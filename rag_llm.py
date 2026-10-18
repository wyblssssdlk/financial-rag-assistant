"""Configured model calls shared by the original CLI and its web presentation."""
import os
import time


class UserError(Exception):
    """A message safe to display to the user."""


class ModelRateLimitError(UserError):
    """Model provider returned HTTP 429 after bounded retries."""


def configure_cloud_network():
    """Use direct TLS for DashScope in this process; leave system settings alone."""
    if os.getenv('DASHSCOPE_USE_SYSTEM_PROXY') == '1':
        return
    exclusions = [host.strip() for host in (os.getenv('no_proxy') or os.getenv('NO_PROXY', '')).split(',') if host.strip()]
    for host in ('dashscope.aliyuncs.com', '127.0.0.1', 'localhost'):
        if host not in exclusions:
            exclusions.append(host)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(exclusions)


def get_answer_model():
    return os.getenv('LLM_MODEL', '').strip() or 'qwen-flash'


def call_answer_model(generate, prompt):
    model = get_answer_model()
    options = {'enable_thinking': False} if model in ('qwen-flash', 'qwen-plus', 'qwen-plus-latest') else {}
    for attempt in range(3):
        response = generate(model=model, messages=[{'role': 'user', 'content': prompt}],
                            result_format='message', timeout=90, **options)
        if response.status_code == 200:
            return response.output.choices[0]['message']['content']
        code = str(getattr(response, 'code', '') or '')
        if response.status_code in (429, 503) and (not code or code.startswith('Throttling') or response.status_code == 503) and attempt < 2:
            time.sleep(5 * (2 ** attempt))
            continue
        if response.status_code == 429:
            raise ModelRateLimitError('模型调用受限（429），请稍后重试；若持续出现，请检查阿里云百炼的模型限流与额度配置。')
        if response.status_code == 401:
            raise UserError('API 密钥认证失败，请检查 DASHSCOPE_API_KEY 配置后重启。')
        if response.status_code == 403:
            raise UserError('没有回答模型的调用权限，请检查阿里云百炼的模型权限与账户状态。')
        raise UserError('模型服务暂时不可用，请检查网络或稍后重试。')


def safe_error(error):
    """Turn an exception into a short Chinese message; never echo provider details or keys."""
    if isinstance(error, UserError):
        return str(error)
    if isinstance(error, ModuleNotFoundError):
        return f'缺少运行依赖 {error.name}，请运行 install.bat（或 pip install -r requirements.txt）后重新启动。'
    if 'Cannot open header file' in str(error):
        return '向量索引文件无法读取：Windows 下索引路径不能含中文，请把项目放在纯英文路径下，或使用网页端（会自动迁移索引）。'
    if type(error).__name__ == 'SSLError':
        return '模型服务的 HTTPS 连接失败，请检查网络；若 .env 中设置了 DASHSCOPE_USE_SYSTEM_PROXY=1，请确认代理可以正常访问 dashscope.aliyuncs.com。'
    if type(error).__name__ in ('Timeout', 'ConnectTimeout', 'ReadTimeout', 'ConnectionError'):
        return '模型服务连接失败或超时，请检查网络后重试。'
    if 'status_code: 401' in str(error) or 'InvalidApiKey' in str(error):
        return 'API 密钥认证失败，请检查项目的 DASHSCOPE_API_KEY 配置后重启。'
    return '处理未完成，请检查网络、模型服务和项目运行环境，再重试。'
