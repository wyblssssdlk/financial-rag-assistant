"""Local presentation layer for the existing financial RAG assistant."""
import importlib
import importlib.util
import hashlib
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from rag_llm import UserError, ModelRateLimitError, get_answer_model, call_answer_model, safe_error, configure_cloud_network

ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass


configure_cloud_network()


def find_content_files(root: Path):
    files = {}
    for folder in ('processed', 'pdf_processed'):
        for path in sorted((root / folder).rglob('*_content_list.json')):
            files.setdefault(path.name, str(path))
    return list(files.values())


def prepare_database_path(root: Path):
    """Windows HNSW cannot open Chinese paths. Preserve the source database."""
    source = root / 'my_chroma_db'
    if os.name != 'nt' or str(source).isascii():
        return source
    base = Path(os.getenv('CHENGMING_DATA_DIR') or str(Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'ChengmingFinancialAssistant')).expanduser().absolute()
    if not str(base).isascii():
        raise UserError('索引存储路径需要使用英文字符，请在 .env 配置 CHENGMING_DATA_DIR，例如 D:/ChengmingData。')
    project_id = hashlib.sha256(str(root.resolve()).casefold().encode('utf-8')).hexdigest()[:16]
    destination = base / 'databases' / project_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sqlite = source / 'chroma.sqlite3'
    if source_sqlite.exists():
        # Copy only a nonempty database; empty metadata can have obsolete HNSW
        # files left from an earlier clear/rebuild. Fresh indexing is safer there.
        try:
            with closing(sqlite3.connect(source_sqlite.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
                count = connection.execute('select count(*) from embeddings').fetchone()[0]
                if count:
                    # This generated scratch directory stays within the chosen
                    # ASCII data directory; only scratch files are cleaned up.
                    with tempfile.TemporaryDirectory(prefix='import-', dir=destination.parent) as scratch:
                        stage = Path(scratch) / 'database'
                        stage.mkdir()
                        for item in source.iterdir():
                            if item.is_dir():
                                shutil.copytree(item, stage / item.name)
                        with closing(sqlite3.connect(stage / 'chroma.sqlite3')) as target:
                            connection.backup(target)
                        stage.rename(destination)
        except sqlite3.Error:
            raise UserError('已有知识库数据库无法读取，请保留原文件并检查数据库备份。') from None
    destination.mkdir(parents=True, exist_ok=True)
    return destination


class ExistingRAGBackend:
    def __init__(self, root):
        self.root = root
        self.rag = None
        self.qa = None

    def initialize(self, report):
        report('正在加载检索组件…')
        # Imports of MinerU and Torch in the core are delayed until PDF parsing.
        spec = importlib.util.spec_from_file_location('financial_assistant_core', self.root / 'financial_assistant.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        database_path = prepare_database_path(self.root)
        self.rag = module.MinerURAGSystem(persist_directory=str(database_path))
        count = self.rag.collection.count()
        if count == 0:
            files = find_content_files(self.root)
            if not files:
                pdfs = list((self.root / 'my_pdfs').glob('*.pdf'))
                if not pdfs:
                    raise UserError('请先把财报 PDF 放入项目的 my_pdfs 文件夹。')
                report('正在解析财报，首次处理可能需要几分钟…')
                self.rag.process_documents(str(self.root / 'my_pdfs'), str(self.root / 'processed'))
                files = find_content_files(self.root)
            if not files:
                raise UserError('未找到财报解析结果，请检查 MinerU 输出目录。')
            report('正在整理财报文字与表格…')
            documents, metadatas, ids = self.rag.chunk_and_embed(files)
            if not documents:
                raise UserError('财报解析结果为空，暂时不能建立知识库。')
            # Calculate every embedding before persisting: API failure cannot leave
            # a half-built collection mistaken for a complete database on retry.
            embeddings = []
            for start in range(0, len(documents), 16):
                stop = min(start + 16, len(documents))
                report(f'正在建立知识库 · {stop} / {len(documents)} 个片段')
                embeddings.extend(self.rag.embedding_model.embed_documents(documents[start:stop]))
            self.rag.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            count = self.rag.collection.count()
        report(f'已载入 {count} 个财报片段')

        self.qa = module.RAGQASystem(self.rag)
        return count

    def ask(self, question):
        count = self.rag.collection.count()
        if not count:
            raise UserError('知识库已变为空，请重新初始化。')
        # One direct call into the original script's complete QA pipeline.
        answer, results = self.qa.ask_question(question)
        if not isinstance(results, dict):
            results = {}
        docs = results.get('documents', [[]])[0]
        metadata = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        sources = []
        for index, text in enumerate(docs):
            item = metadata[index] if index < len(metadata) and metadata[index] else {}
            sources.append({'text': text, 'source': item.get('source', '财报'),
                            'distance': distances[index] if index < len(distances) else None,
                            'page_idx': item.get('page_idx')})
        return {'answer': answer, 'sources': sources}


class AssistantService:
    def __init__(self, root, backend_factory=ExistingRAGBackend, key_checker=None):
        self.root = Path(root)
        self.backend_factory = backend_factory
        self.key_checker = key_checker or (lambda: bool(os.getenv('DASHSCOPE_API_KEY', '').strip()))
        self.backend = None
        self.lock = threading.Lock()
        self.operation_lock = threading.Lock()
        self.worker = None
        self.state = 'idle'
        self.message = '初始化后，即可开始分析财报'
        self.chunk_count = 0

    def snapshot(self):
        with self.lock:
            return {'state': self.state, 'message': self.message, 'chunk_count': self.chunk_count,
                    'configured': self.key_checker(),
                    'files': [{'name': p.name, 'size': p.stat().st_size} for p in sorted((self.root / 'my_pdfs').glob('*.pdf'))],
                    'model': {'qwen-turbo': 'Qwen Turbo', 'qwen-flash': 'Qwen Flash', 'qwen-plus': 'Qwen Plus'}.get(get_answer_model(), get_answer_model()),
                    'processed_count': len(find_content_files(self.root))}

    def report(self, message):
        with self.lock:
            self.message = message

    def initialize(self):
        if not self.key_checker():
            raise HTTPException(400, '尚未配置 DASHSCOPE_API_KEY，请在项目 .env 中配置后重新启动。')
        if not self.operation_lock.acquire(blocking=False):
            raise HTTPException(409, '正在处理上一项任务，请稍后再试。')
        with self.lock:
            if self.state == 'ready':
                self.operation_lock.release()
                return self.snapshot_unlocked()
            self.state = 'initializing'
            self.message = '正在准备财报知识库…'
        self.worker = threading.Thread(target=self._initialize, daemon=True)
        self.worker.start()
        return {'state': 'initializing'}

    def snapshot_unlocked(self):
        return {'state': self.state, 'message': self.message, 'chunk_count': self.chunk_count}

    def _initialize(self):
        try:
            backend = self.backend_factory(self.root)
            count = backend.initialize(self.report)
            if not count:
                raise UserError('知识库没有可用的财报片段。')
            with self.lock:
                self.backend = backend
                self.chunk_count = count
                self.state = 'ready'
                self.message = f'知识库就绪 · {count} 个财报片段'
        except Exception as error:
            with self.lock:
                self.state = 'error'
                self.message = safe_error(error)
        finally:
            self.operation_lock.release()

    def ask(self, question):
        if not self.operation_lock.acquire(blocking=False):
            raise HTTPException(409, '正在处理上一项任务，请稍后再试。')
        try:
            with self.lock:
                if self.state != 'ready':
                    raise HTTPException(409, '请先初始化财报知识库。')
            started = time.perf_counter()
            result = self.backend.ask(question)
            return {**result, 'elapsed_seconds': round(time.perf_counter() - started, 2)}
        except HTTPException:
            raise
        except ModelRateLimitError as error:
            raise HTTPException(429, safe_error(error)) from None
        except Exception as error:
            raise HTTPException(502, safe_error(error)) from None
        finally:
            self.operation_lock.release()

    def wait(self, timeout=None):
        if self.worker:
            self.worker.join(timeout)


class Question(BaseModel):
    question: str = Field(max_length=2000)

    @field_validator('question')
    @classmethod
    def nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('请输入一个财报问题')
        return value


def create_app(service=None):
    service = service or AssistantService(ROOT)
    app = FastAPI(title='澄明 · 财报研究助手', docs_url=None, redoc_url=None)

    @app.get('/api/status')
    def status():
        return service.snapshot()

    @app.post('/api/initialize')
    def initialize():
        return service.initialize()

    @app.post('/api/ask')
    def ask(question: Question):
        return service.ask(question.question)

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    if (ROOT / 'static').is_dir():
        app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, access_log=False)
