import importlib.util
from contextlib import closing
import os
import sqlite3
import threading
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('web_app', ROOT / 'web_app.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class FakeBackend:
    def __init__(self, count=3):
        self.count = count
        self.builds = 0
        self.entered = threading.Event()
        self.release = None

    def initialize(self, report):
        self.entered.set()
        if self.release:
            self.release.wait(2)
        report('正在读取知识库')
        if not self.count:
            self.builds += 1
            self.count = 4
        return self.count

    def ask(self, question):
        return {'answer': '净利润为财报披露金额。', 'sources': [{'text': '<script>danger</script>', 'source': '年报.pdf', 'distance': .3}]}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.backend = FakeBackend()
        self.service = web.AssistantService(Path(self.temp.name), backend_factory=lambda path: self.backend, key_checker=lambda: True)
        self.client = TestClient(web.create_app(self.service))

    def tearDown(self):
        self.service.wait(3)
        self.temp.cleanup()

    def ready(self):
        self.client.post('/api/initialize')
        self.service.wait(3)

    def test_empty_question_is_rejected(self):
        self.assertEqual(self.client.post('/api/ask', json={'question': '   '}).status_code, 422)

    def test_question_requires_ready_database(self):
        self.assertEqual(self.client.post('/api/ask', json={'question': '净利润？'}).status_code, 409)

    def test_missing_key_is_actionable_and_not_leaked(self):
        self.service.key_checker = lambda: False
        response = self.client.post('/api/initialize')
        self.assertEqual(response.status_code, 400)
        self.assertIn('DASHSCOPE_API_KEY', response.json()['detail'])

    def test_existing_database_is_reused(self):
        self.ready()
        self.assertEqual(self.backend.builds, 0)
        status = self.client.get('/api/status').json()
        self.assertEqual((status['state'], status['chunk_count']), ('ready', 3))

    def test_empty_database_is_built(self):
        self.backend.count = 0
        self.ready()
        self.assertEqual(self.backend.builds, 1)
        self.assertEqual(self.client.get('/api/status').json()['chunk_count'], 4)

    def test_concurrent_initialization_is_not_duplicated(self):
        self.backend.release = threading.Event()
        self.client.post('/api/initialize')
        self.backend.entered.wait(1)
        self.assertEqual(self.client.post('/api/initialize').status_code, 409)
        self.assertEqual(self.client.post('/api/ask', json={'question': '营业收入？'}).status_code, 409)
        self.backend.release.set()

    def test_answer_preserves_sources_and_has_timing(self):
        self.ready()
        response = self.client.post('/api/ask', json={'question': '净利润？'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sources'][0]['source'], '年报.pdf')
        self.assertGreaterEqual(response.json()['elapsed_seconds'], 0)

    def test_internal_exception_does_not_leak_secret(self):
        self.backend.initialize = lambda report: (_ for _ in ()).throw(RuntimeError('secret-key-123'))
        self.ready()
        status = self.client.get('/api/status').json()
        self.assertEqual(status['state'], 'error')
        self.assertNotIn('secret-key-123', str(status))

    def test_upstream_rate_limit_is_returned_as_429(self):
        self.ready()
        self.backend.ask = Mock(side_effect=web.ModelRateLimitError('模型调用受限（429）'))
        response = self.client.post('/api/ask', json={'question': '净利润？'})
        self.assertEqual(response.status_code, 429)
        self.assertIn('429', response.json()['detail'])

    def test_web_backend_calls_original_ask_question_once(self):
        backend = web.ExistingRAGBackend(Path(self.temp.name))
        backend.rag = SimpleNamespace(collection=SimpleNamespace(count=lambda: 1), query_documents=Mock(side_effect=RuntimeError('网页不应单独重复检索')))
        results = {'documents': [['完整财报正文']], 'metadatas': [[{'source': '年报.pdf'}]], 'distances': [[.25]]}
        backend.qa = SimpleNamespace(ask_question=Mock(return_value=('原脚本生成的回答',results)))
        result = backend.ask('公司主要业务？')
        backend.qa.ask_question.assert_called_once_with('公司主要业务？')
        backend.rag.query_documents.assert_not_called()
        self.assertEqual(result['answer'], '原脚本生成的回答')
        self.assertEqual(result['sources'][0]['text'], '完整财报正文')

    def test_processed_files_are_found_recursively_without_duplicates(self):
        path = Path(self.temp.name) / 'processed' / 'report' / 'auto'
        path.mkdir(parents=True)
        (path / 'report_content_list.json').write_text('[]')
        duplicate = Path(self.temp.name) / 'pdf_processed' / 'auto'
        duplicate.mkdir(parents=True)
        (duplicate / 'report_content_list.json').write_text('[]')
        self.assertEqual(len(web.find_content_files(Path(self.temp.name))), 1)


class NetworkConfigurationTests(unittest.TestCase):
    def test_dashscope_bypasses_broken_system_proxy_and_keeps_other_exclusions(self):
        with patch.dict(os.environ, {'NO_PROXY': 'localhost', 'no_proxy': 'localhost', 'DASHSCOPE_USE_SYSTEM_PROXY': '0'}):
            web.configure_cloud_network()
            self.assertIn('dashscope.aliyuncs.com', os.environ['NO_PROXY'])
            self.assertIn('localhost', os.environ['NO_PROXY'])
            self.assertEqual(os.environ['NO_PROXY'], os.environ['no_proxy'])

    def test_explicit_proxy_preference_is_preserved(self):
        with patch.dict(os.environ, {'NO_PROXY': 'localhost', 'no_proxy': 'localhost', 'DASHSCOPE_USE_SYSTEM_PROXY': '1'}):
            web.configure_cloud_network()
            self.assertEqual(os.environ['NO_PROXY'], 'localhost')


class DatabasePathTests(unittest.TestCase):
    def test_unicode_project_uses_ascii_storage_without_changing_original(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / '中文项目'
            original = root / 'my_chroma_db'
            original.mkdir(parents=True)
            with closing(sqlite3.connect(original / 'chroma.sqlite3')) as db:
                db.execute('create table embeddings(id integer)')
                db.execute('insert into embeddings values (1)')
                db.commit()
            (original / 'segment').mkdir()
            (original / 'segment' / 'header.bin').write_bytes(b'original-index')
            with patch.dict(os.environ, {'CHENGMING_DATA_DIR': str(Path(folder) / 'ascii-data')}):
                destination = web.prepare_database_path(root)
                self.assertTrue(str(destination).isascii())
                self.assertNotEqual(destination, original)
                with closing(sqlite3.connect(destination / 'chroma.sqlite3')) as db:
                    self.assertEqual(db.execute('select count(*) from embeddings').fetchone()[0], 1)
                self.assertEqual((destination / 'segment' / 'header.bin').read_bytes(), b'original-index')
                self.assertEqual((original / 'segment' / 'header.bin').read_bytes(), b'original-index')
                self.assertEqual(web.prepare_database_path(root), destination)

    def test_empty_database_does_not_copy_stale_index(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / '中文项目'
            original = root / 'my_chroma_db'
            original.mkdir(parents=True)
            with closing(sqlite3.connect(original / 'chroma.sqlite3')) as db:
                db.execute('create table embeddings(id integer)')
                db.commit()
            (original / 'stale-index').mkdir()
            with patch.dict(os.environ, {'CHENGMING_DATA_DIR': str(Path(folder) / 'ascii-data')}):
                destination = web.prepare_database_path(root)
                self.assertFalse((destination / 'stale-index').exists())
                self.assertTrue((original / 'stale-index').exists())

    def test_ascii_project_keeps_existing_location(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'ascii-project'
            self.assertEqual(web.prepare_database_path(root), root / 'my_chroma_db')

    def test_native_index_failure_is_actionable(self):
        message = web.safe_error(RuntimeError('Cannot open header file'))
        self.assertIn('索引', message)
        self.assertNotEqual(message, '处理未完成，请检查网络、模型服务和项目运行环境，再重试。')


class ModelRetryTests(unittest.TestCase):
    def test_transient_rate_limit_is_retried(self):
        limited = SimpleNamespace(status_code=429, code='Throttling.RateQuota')
        success = SimpleNamespace(status_code=200, output=SimpleNamespace(choices=[{'message': {'content': '已回答'}}]))
        generate = Mock(side_effect=[limited, success])
        with patch.object(web.time, 'sleep') as sleep:
            self.assertEqual(web.call_answer_model(generate, '测试问题'), '已回答')
            self.assertEqual(generate.call_count, 2)
            sleep.assert_called_once_with(5)

    def test_persistent_rate_limit_is_actionable_and_bounded(self):
        generate = Mock(return_value=SimpleNamespace(status_code=429, code='Throttling.RateQuota'))
        with patch.object(web.time, 'sleep'):
            with self.assertRaisesRegex(web.UserError, '429'):
                web.call_answer_model(generate, '测试问题')
        self.assertEqual(generate.call_count, 3)

    def test_authentication_failure_does_not_retry(self):
        generate = Mock(return_value=SimpleNamespace(status_code=401, code='InvalidApiKey'))
        with patch.object(web.time, 'sleep') as sleep:
            with self.assertRaisesRegex(web.UserError, '密钥'):
                web.call_answer_model(generate, '测试问题')
            sleep.assert_not_called()
        self.assertEqual(generate.call_count, 1)

    def test_configured_model_is_used_in_request_and_ui(self):
        success = SimpleNamespace(status_code=200, output=SimpleNamespace(choices=[{'message': {'content': '已回答'}}]))
        generate = Mock(return_value=success)
        with patch.dict(os.environ, {'LLM_MODEL': 'qwen-flash'}):
            self.assertEqual(web.call_answer_model(generate, '测试问题'), '已回答')
            self.assertEqual(generate.call_args.kwargs['model'], 'qwen-flash')
            self.assertEqual(generate.call_args.kwargs['enable_thinking'], False)
            service = web.AssistantService(ROOT, key_checker=lambda: True)
            self.assertEqual(service.snapshot()['model'], 'Qwen Flash')

    def test_empty_model_configuration_uses_default(self):
        with patch.dict(os.environ, {'LLM_MODEL': '   '}):
            self.assertEqual(web.get_answer_model(), 'qwen-flash')


if __name__ == '__main__':
    unittest.main()
