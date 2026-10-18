import unittest
import os
import contextlib
import io
from unittest.mock import Mock, patch
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_assistant import MinerURAGSystem, RAGQASystem, dashscope


class InMemoryCollection:
    """Small deterministic vector store; no cloud requests or persistent writes."""
    metadata = {}

    def __init__(self, documents):
        self.records = documents

    def count(self):
        return len(self.records)

    def get(self, ids=None, include=None):
        ids = [key for key in (ids or list(self.records)) if key in self.records]
        result = {'ids': ids}
        if 'documents' in include:
            result['documents'] = [self.records[key]['text'] for key in ids]
        if 'metadatas' in include:
            result['metadatas'] = [self.records[key]['metadata'] for key in ids]
        if 'embeddings' in include:
            result['embeddings'] = [self.records[key]['embedding'] for key in ids]
        return result

    def query(self, query_embeddings, n_results, include):
        vector = query_embeddings[0]
        ranked = sorted(self.records, key=lambda key: sum((a-b)**2 for a,b in zip(vector,self.records[key]['embedding'])))[:n_results]
        return {'ids': [ranked], 'documents': [[self.records[key]['text'] for key in ranked]],
                'metadatas': [[self.records[key]['metadata'] for key in ranked]],
                'distances': [[sum((a-b)**2 for a,b in zip(vector,self.records[key]['embedding'])) for key in ranked]]}


def record(text, distance=1, kind='text', source='report_content_list.json'):
    return {'text': text, 'embedding': [distance, 0],
            'metadata': {'source': source, 'chunk_type': kind, 'page_idx': 8}}


class RetrievalTests(unittest.TestCase):
    def core(self, records):
        system = MinerURAGSystem.__new__(MinerURAGSystem)
        system.collection = InMemoryCollection(records)
        system.embedding_model = Mock()
        system.embedding_model.embed_query.return_value = [0,0]
        system.embedding_model.embed_documents.return_value = [[0,0]]
        return system

    def test_title_hit_includes_its_following_asset_tables(self):
        system = self.core({
            'report_content_list.json_chunk_0': record('## 二、主要资产重大变化情况',.1),
            'report_content_list.json_chunk_1': record('## 1、主要资产重大变化情况',.05),
            'report_content_list.json_chunk_2': record('<table><tr><td>主要资产</td><td>重大变化说明</td></tr><tr><td>货币资金</td><td>增加36.18%</td></tr></table>',2,'table'),
            'report_content_list.json_chunk_3': record('<table><tr><td>主要资产</td><td>重大变化说明</td></tr><tr><td>固定资产</td><td>无重大变化</td></tr></table>',3,'table'),
            'report_content_list.json_chunk_4': record('## 2、主要境外资产情况',1),
        })
        result=system.query_documents('主要资产重大变化情况',n_results=2)
        self.assertIn('货币资金',result['documents'][0][0])
        self.assertIn('36.18%',result['documents'][0][0])
        self.assertIn('固定资产',result['documents'][0][0])
        self.assertNotIn('主要境外资产',result['documents'][0][0])

    def test_exact_business_phrase_beats_unrelated_short_vector_title(self):
        system=self.core({
            'report_content_list.json_chunk_0': record('## 一、报告期内公司从事的主要业务\n开发、生产家用电器，设计制造模具，并提供售后服务。',3),
            'report_content_list.json_chunk_1': record('## 二、主营业务分析',.01),
            'report_content_list.json_chunk_2': record('## 1、概述\n年度营业收入增加11.84%。',1),
        })
        result=system.query_documents('报告期内公司从事的主要业务',n_results=1)
        self.assertIn('开发、生产家用电器',result['documents'][0][0])

    def test_query_uses_query_embedding(self):
        system=self.core({'report_content_list.json_chunk_0':record('净利润为34,325,162.34元。')})
        system.query_documents('一季度净利润',n_results=1)
        system.embedding_model.embed_query.assert_called_once_with('一季度净利润')
        system.embedding_model.embed_documents.assert_not_called()

    def test_context_stays_within_source_document(self):
        system=self.core({
            'a_content_list.json_chunk_0':record('## 主要资产重大变化情况',.01,source='a_content_list.json'),
            'a_content_list.json_chunk_1':record('货币资金增加36.18%。',1,source='a_content_list.json'),
            'b_content_list.json_chunk_1':record('另一公司货币资金减少99%。',2,source='b_content_list.json'),
        })
        result=system.query_documents('主要资产重大变化情况',n_results=1)
        self.assertIn('增加36.18%',result['documents'][0][0])
        self.assertNotIn('另一公司',result['documents'][0][0])

    def test_overlapping_title_and_table_hits_do_not_repeat_same_context(self):
        system=self.core({
            'report_content_list.json_chunk_0':record('## 主要资产重大变化情况',.02),
            'report_content_list.json_chunk_1':record('## 1、主要资产重大变化情况',.01),
            'report_content_list.json_chunk_2':record('<table><tr><td>主要资产</td></tr><tr><td>货币资金增加36.18%</td></tr></table>',.03,'table'),
        })
        result=system.query_documents('主要资产重大变化情况',n_results=5)
        contexts=result['documents'][0]
        self.assertEqual(sum('货币资金' in text for text in contexts),1)

    def test_results_remain_compatible_with_web_and_cli(self):
        system=self.core({'report_content_list.json_chunk_0':record('一季度净利润34,325,162.34元。')})
        result=system.query_documents('净利润',n_results=1)
        self.assertEqual(set(result),{'ids','documents','metadatas','distances'})
        for key in result:
            self.assertEqual(len(result[key]),1)
            self.assertEqual(len(result[key][0]),1)
        self.assertIsInstance(result['distances'][0][0],float)

    def test_sibling_heading_with_body_is_not_attached_to_asset_title(self):
        system=self.core({
            'report_content_list.json_chunk_0':record('## 二、主要资产重大变化情况',.01),
            'report_content_list.json_chunk_1':record('## 三、核心竞争力分析\n研发实力强，产品创新能力突出。',.02),
        })
        result=system.query_documents('主要资产重大变化情况',n_results=2)
        self.assertFalse(any('主要资产重大变化情况' in text and '核心竞争力分析' in text for text in result['documents'][0]))

    def test_preceding_sibling_sections_are_not_prepended(self):
        system=self.core({
            'report_content_list.json_chunk_0':record('## 六、重大资产和股权出售',1),
            'report_content_list.json_chunk_1':record('## 1、出售重大资产情况 □适用√不适用',2),
            'report_content_list.json_chunk_2':record('## 2、出售重大股权情况 □适用√不适用',3),
            'report_content_list.json_chunk_3':record('## 七、主要控股参股公司分析\n主要子公司生产电器，并提供售后服务。',.01),
        })
        result=system.query_documents('主要控股参股公司分析',n_results=1)
        self.assertIn('主要子公司',result['documents'][0][0])
        self.assertNotIn('出售重大',result['documents'][0][0])

    def test_partial_overlap_keeps_unique_table_rows(self):
        records={}
        for i in range(5):
            term='revenue 100' if i==0 else 'profit 123' if i==4 else f'asset {i}'
            text='<table><tr><td>item</td><td>value</td></tr><tr><td>'+term+'</td><td>'+('x'*550)+'</td></tr></table>'
            records[f'report_content_list.json_chunk_{i}']=record(text,.01+i*.2,'table')
        system=self.core(records)
        result=system.query_documents('revenue profit',n_results=5)
        self.assertTrue(any('revenue 100' in text for text in result['documents'][0]))
        self.assertTrue(any('profit 123' in text for text in result['documents'][0]))


class CoreModelTests(unittest.TestCase):
    def test_original_core_model_call_respects_configured_flash(self):
        response=SimpleNamespace(status_code=200,output=SimpleNamespace(choices=[{'message':{'content':'已回答'}}]))
        qa=RAGQASystem.__new__(RAGQASystem)
        with patch.dict(os.environ,{'LLM_MODEL':'qwen-flash'}), patch.object(dashscope.Generation,'call',return_value=response) as generate, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qa._call_llm('测试问题'),'已回答')
        self.assertEqual(generate.call_args.kwargs['model'],'qwen-flash')
        self.assertEqual(generate.call_args.kwargs['enable_thinking'],False)


if __name__=='__main__':
    unittest.main()
