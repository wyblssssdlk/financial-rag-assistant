"""金融财报 RAG 助手核心：MinerU 解析 PDF → 分块 → DashScope 向量化 → Chroma 混合检索 → 通义千问回答。

用法：
    python financial_assistant.py                        # 演示流程：解析/建库（会清空旧集合）并提问一次
    python financial_assistant.py search --question ...  # 只读检索，输出 JSON
    python financial_assistant.py ask --question ...     # 只读检索 + 生成回答，输出 JSON
"""
import glob
import os
import json
import sys
import math
import re
from collections import Counter
from datetime import datetime
import chromadb
from chromadb.config import Settings
import dashscope

from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from pathlib import Path
from dotenv import load_dotenv
from rag_retrieval import retrieve_with_context
from rag_llm import call_answer_model, get_answer_model, safe_error, configure_cloud_network

load_dotenv()

# 提示词中引用的财报名称；更换财报时在 .env 中设置 REPORT_TITLE。
REPORT_TITLE = os.getenv('REPORT_TITLE', '厦门灿坤实业股份有限公司2019年年度报告')

# 内容提取相关
class MinerURAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedding_model = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v2"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection("documents")

    # 加载文档，进行pdf的文档预处理
    def process_documents(self, pdf_directory, output_dir="./processed"):
        # 已解析的财报可直接检索；仅解析新 PDF 时才加载重型依赖。
        import torch
        import doclayout_yolo.nn.tasks
        from mineru.cli.common import do_parse, read_fn
        torch.serialization.add_safe_globals([doclayout_yolo.nn.tasks.YOLOv10DetectionModel])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        processed_files = []
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                input_path = os.path.join(pdf_directory, filename)
                # output_path = os.path.join(output_dir, filename.replace('.pdf', ''))
                output_path = output_dir
                pdf_file_name = Path(input_path).stem
                pdf_bytes = read_fn(input_path)
                do_parse(
                    output_dir=output_path,
                    pdf_file_names=[pdf_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],
                    backend="pipeline"
                )
                # MinerU 输出可能包含 PDF 文件名目录。
                matches = sorted(Path(output_path).rglob(f"{pdf_file_name}_content_list.json"))
                processed_files.extend(str(path) for path in matches)
        return processed_files

    # 切块
    def _split_into_chunks(self, content_data, chunk_size, overlap):
        chunks = []
        current_chunk = []
        current_length = 0

        for item in content_data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if not text:
                    continue
                if item.get('text_level', 0) > 0:
                    if current_chunk:
                        chunks.append(
                            {'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text', 'has_title': True}})
                    current_chunk = [f"## {text}"]
                    current_length = len(text)
                elif current_length + len(text) > chunk_size and current_chunk:
                    chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
                    current_chunk = [text]
                    current_length = len(text)
                else:
                    current_chunk.append(text)
                    current_length += len(text)

            elif item.get('type') == 'table':
                if current_chunk:
                    chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
                    current_chunk = []
                    current_length = 0

                table_body = item.get('table_body', '')
                if table_body:
                    if len(table_body) <= chunk_size:
                        chunks.append({
                            'text': table_body,
                            'metadata': {
                                'chunk_type': 'table',
                                'page_idx': item.get('page_idx', 0),
                                'has_caption': len(item.get('table_caption', [])) > 0,
                                'has_footnote': len(item.get('table_footnote', [])) > 0
                            }
                        })
                    else:
                        table_chunks = self._split_table_into_chunks(table_body, chunk_size)
                        for i, chunk in enumerate(table_chunks):
                            chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'chunk_type': 'table',
                                    'page_idx': item.get('page_idx', 0),
                                    'table_chunk_index': i,
                                    'total_table_chunks': len(table_chunks)
                                }
                            })
        if current_chunk:
            chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
        return chunks

    # 表格切块
    def _split_table_into_chunks(self, table_html, chunk_size):
        chunks = []
        if table_html.startswith('<table>'):
            table_content = table_html[7:-8]
        else:
            table_content = table_html
        rows = table_content.split('<tr>')
        table_head = rows[1]
        current_chunk = '<table>' + table_head

        current_chunk_size = len(current_chunk)
        for row in rows[2:]:
            row_content = '<tr>' + row
            if current_chunk_size + len(row_content) > chunk_size and current_chunk != '<table>':
                current_chunk += '</table>'
                chunks.append(current_chunk)
                current_chunk = '<table>' + table_head + row_content
                current_chunk_size = len(current_chunk)
            else:
                current_chunk += row_content
                current_chunk_size += len(row_content)

        if current_chunk != '<table>' + table_head:
            current_chunk += '</table>'
            chunks.append(current_chunk)
        return chunks

    # 向量化
    def chunk_and_embed(self, content_files, chunk_size=512, overlap=50):
        documents = []
        metadatas = []
        ids = []
        print()
        for file_path in content_files:
            print(f"正在处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            doc_metadata = {
                'source': os.path.basename(file_path),
                'file_type': 'pdf',
                'processing_date': datetime.now().isoformat()
            }
            text_chunks = self._split_into_chunks(content_data, chunk_size, overlap)
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk['text'])
                metadatas.append({**doc_metadata, **chunk['metadata']})
                ids.append(f"{os.path.basename(file_path)}_chunk_{i}")
        return documents, metadatas, ids

    # 存储
    def build_vector_store(self, documents, metadatas, ids):
        if self.collection.count() > 0:
            print(f"集合不为空，正在删除 {self.collection.count()} 个旧文档...")
            existing_ids = self.collection.get(include=[])['ids']
            self.collection.delete(ids=existing_ids)
            print("旧文档删除完毕。")
        else:
            print("集合为空，无需删除。")

        embeddings = self.embedding_model.embed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"成功存储 {len(documents)} 个文档块到向量数据库")

    # 检索
    def query_documents(self, query_text, n_results=5):
        query_embedding = self.embedding_model.embed_query(query_text)
        return retrieve_with_context(self.collection, query_text, query_embedding, n_results)


# 集成功能完善的 RAGQASystem
class RAGQASystem:
    def __init__(self, mineru_rag_system):
        self.rag_system = mineru_rag_system
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    def _build_prompt(self, question, context_docs):
        context_text = "\n\n".join([f"文档片段 {i + 1}:\n{doc}" for i, doc in enumerate(context_docs)])
        prompt = f"""请你扮演一个专业的财务报告分析师。
                    根据下面提供的几段从《{REPORT_TITLE}》中摘录的文字，
                    严谨且仅依据这些信息来回答用户的问题。
                    如果提供的信息不足以回答问题，请明确告知"根据现有信息无法回答"。

                    ---
                    相关文档内容：
                    {context_text}
                    ---

                    用户问题：{question}
                    """
        return prompt

    def _call_llm(self, prompt):
        print(f"\n--- 正在向DashScope发送请求（{get_answer_model()}）... ---")
        answer = call_answer_model(dashscope.Generation.call, prompt)
        print("--- LLM响应成功 ---")
        return answer

    def generate_answer(self, question, context):
        prompt = self._build_prompt(question, context)
        print(f"\n--- 正在根据 {len(context)} 个财报片段生成回答 ---")
        answer = self._call_llm(prompt)
        return answer, context

    def ask_question(self, question):
        results = self.rag_system.query_documents(question)
        if not results or not results.get('documents') or not results['documents'][0]:
            return "抱歉，没有找到相关的文档信息。", []

        answer, context = self.generate_answer(question, results['documents'][0])
        return answer, results



# ==============================================================================
# --- 主调用流程（修复后的版本）---
# ==============================================================================
def _query_cli(argv=None):
    """Read-only search/ask CLI; failures print {"ok": false, "error": 中文原因} and exit 1."""
    try:
        return _run_query_cli(argv)
    except Exception as error:
        print(json.dumps({'ok': False, 'error': safe_error(error)}, ensure_ascii=False))
        return 1


def _run_query_cli(argv=None):
    """Read-only search/ask CLI for an existing index or MinerU parse files."""
    import argparse

    parser = argparse.ArgumentParser(description='查询已有财报知识库，不会解析 PDF 或重建索引。')
    parser.add_argument('command', choices=['search', 'ask'])
    parser.add_argument('--question', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--db-dir', default='my_chroma_db')
    args = parser.parse_args(argv)
    if not args.question.strip() or not 1 <= args.top_k <= 20:
        parser.error('question 不能为空，top-k 必须在 1 到 20 之间')

    project = Path(__file__).resolve().parent
    os.chdir(project)
    load_dotenv(project / '.env')
    configure_cloud_network()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    evidence = []
    retrieval_mode = None
    database = Path(args.db_dir)
    if not database.is_absolute():
        database = project / database
    if (database / 'chroma.sqlite3').is_file():
        try:
            client = chromadb.PersistentClient(
                path=str(database), settings=Settings(anonymized_telemetry=False))
            collection = client.get_collection('documents')
            count = collection.count()
        except Exception:
            collection, count = None, 0
        if count:
            if not os.getenv('DASHSCOPE_API_KEY'):
                parser.error('向量检索需要配置 DASHSCOPE_API_KEY')
            rag = MinerURAGSystem(persist_directory=str(database))
            results = rag.query_documents(args.question, min(args.top_k, count))
            docs = (results.get('documents') or [[]])[0]
            metas = (results.get('metadatas') or [[]])[0]
            distances = (results.get('distances') or [[]])[0]
            evidence = [dict(text=doc,
                             metadata=metas[i] if i < len(metas) else {},
                             distance=distances[i] if i < len(distances) else None)
                        for i, doc in enumerate(docs)]
            retrieval_mode = 'chroma_hybrid'

    if retrieval_mode is None:
        # Offline fallback over existing MinerU output. This never embeds or mutates Chroma.
        files = sorted({p for name in ('processed', 'pdf_processed')
                        for root in [project / name] if root.exists()
                        for p in root.rglob('*_content_list.json')})
        records = []
        for path in files:
            try:
                items = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            for item in items:
                kind = item.get('type')
                if kind == 'table' and item.get('table_body'):
                    text = ' '.join(str(x) for x in item.get('table_caption', [])) + '\n' + str(item['table_body'])
                elif kind == 'text' and item.get('text'):
                    text = str(item['text'])
                else:
                    continue
                records.append({'text': text,
                                'metadata': {'source': path.name, 'page_idx': item.get('page_idx'),
                                             'record_type': kind}})
        def tokens(value):
            result = []
            for part in re.findall(r'[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+', value.casefold()):
                result.extend(part[i:i + 2] for i in range(len(part) - 1)) if re.fullmatch(r'[\u4e00-\u9fff]+', part) else result.append(part)
            return result
        query = set(tokens(args.question))
        counts = [Counter(tokens(row['text'])) for row in records]
        df = Counter(term for row in counts for term in row)
        average = sum(sum(row.values()) for row in counts) / max(len(counts), 1)
        ranked = []
        for record, terms in zip(records, counts):
            length = sum(terms.values())
            score = sum(math.log(1 + (len(records) - df[t] + .5) / (df[t] + .5)) * tf * 2.2 /
                        (tf + 1.2 * (.25 + .75 * length / max(average, 1)))
                        for t in query if (tf := terms[t]))
            normalized_q = re.sub(r'\W', '', args.question).casefold()
            if len(normalized_q) >= 4 and normalized_q in re.sub(r'\W', '', record['text']).casefold():
                score += 3
            if score:
                ranked.append((score, record))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        evidence = [dict(row, distance=None) for _, row in ranked[:args.top_k]]
        retrieval_mode = 'parsed_files_bm25'
        if not files:
            parser.error('Chroma 索引为空，且 processed/pdf_processed 下没有 MinerU 解析文件')

    output = {'ok': True, 'command': args.command, 'question': args.question,
              'retrieval_mode': retrieval_mode, 'evidence': evidence}
    if args.command == 'ask':
        if not evidence:
            output['answer'] = '根据现有信息无法回答。'
        else:
            if not os.getenv('DASHSCOPE_API_KEY'):
                parser.error('ask 生成答案需要配置 DASHSCOPE_API_KEY')
            context = '\n\n'.join(f"证据 {i + 1}: {item['text']} 来源:{item['metadata'].get('source')}"
                                    for i, item in enumerate(evidence))
            prompt = ('请严格依据以下财报证据回答。核对公司、报告年度、期间、单位和口径；'
                      '证据不足时明确说明无法回答，不要编造。引用证据编号和来源。\n'
                      f'问题：{args.question}\n证据：\n{context}')
            output['answer'] = call_answer_model(dashscope.Generation.call, prompt)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    # Agent CLI is read-only. The no-argument demo below retains its original behavior.
    if len(sys.argv) > 1 and sys.argv[1] in {'search', 'ask'}:
        raise SystemExit(_query_cli())

    # 1. 初始化RAG系统（DashScope 请求默认绕过系统代理直连）
    configure_cloud_network()
    rag_system = MinerURAGSystem(persist_directory="./my_chroma_db")

    # 2. 定义路径
    pdf_folder = 'my_pdfs'
    processed_base_dir = './processed'

    # 3. 检查是否有PDF文件可处理
    if not os.path.exists(pdf_folder) or len([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]) == 0:
        print(f"请在当前目录下创建'{pdf_folder}'文件夹并放入PDF文件。")
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
        sys.exit(0)  # 退出，等待用户放入PDF

    # 4. 查找已处理的文件
    processed_files = []
    if os.path.exists(processed_base_dir):
        # 查找所有_content_list.json文件
        pattern = os.path.join(processed_base_dir, '**', '*_content_list.json')
        # print(f"正在查找已处理的文件，搜索模式: {pattern}")
        processed_files = glob.glob(pattern, recursive=True)
        # print(f'processed_files (raw): {processed_files}')
        processed_files = [f for f in processed_files if f]
        print(f'processed_files: {processed_files}')
    # 5. 如果没有已处理的文件，则解析PDF
    if not processed_files:
        print("未找到已处理的文件，开始解析PDF...")
        processed_files = rag_system.process_documents(pdf_directory=pdf_folder)
        print(f"解析完成，生成了 {len(processed_files)} 个内容文件。")
    else:
        print(f"找到 {len(processed_files)} 个已处理的文件，跳过解析步骤。")

    # 6. 文本分块与向量化准备
    print(f"\n开始对 {len(processed_files)} 个文件进行文本分块与向量化...")
    documents, metadatas, ids = rag_system.chunk_and_embed(content_files=processed_files)
    print(f"已将文档分割成 {len(documents)} 个块。")

    # 7. 构建并填充向量数据库
    if documents:
        print("\n开始构建并填充向量数据库...")
        rag_system.build_vector_store(documents, metadatas, ids)

        # 8. 初始化问答系统
        qa_system = RAGQASystem(mineru_rag_system=rag_system)
        print("\n问答系统已准备就绪。")

        # 9. 提出问题
        question = "2019年一季度归属于上市公司股东的净利润？"
        # question = "2019年度归属于上市公司股东的净利润？"

        print(f"\n--- 正在查询问题 --- \n{question}")
        answer, search_results = qa_system.ask_question(question)

        print("\n\n==================== 问答结果 ====================")
        print("--- 最终回答 ---")
        print(answer)
        print("\n--- 检索到的相关信息 ---")
        if search_results and search_results.get('documents') and search_results['documents'][0]:
            for i, doc in enumerate(search_results['documents'][0]):
                distance = search_results['distances'][0][i]
                metadata = search_results['metadatas'][0][i]
                print(f"\n--- 相关片段 {i + 1} (距离: {distance:.4f}) ---")
                print(f"来源: {metadata.get('source')}")
                print("内容:")
                print(doc)
        else:
            print("没有找到相关的文档片段。")
        print("\n==================================================")
    else:
        print("错误：没有成功生成文档块，请检查PDF文件和处理过程。")
