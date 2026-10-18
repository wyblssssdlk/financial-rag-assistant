"""Double-click launcher: start a local server and open the UI when ready."""
import importlib.util
import socket
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def main():
    required = ('fastapi', 'uvicorn', 'dotenv')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print('缺少网页依赖：' + ', '.join(missing))
        print('请先运行 install.bat（或 pip install -r requirements.txt），安装完成后重新启动。')
        return 1
    import uvicorn
    from web_app import app
    # Avoid accidentally opening an unrelated service on our preferred port.
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8765))
            port = 8765
        except OSError:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
    url = f'http://127.0.0.1:{port}'

    def open_when_ready():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(60):
            try:
                with opener.open(url + '/api/status', timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(url)
                        return
            except OSError:
                time.sleep(.5)

    threading.Thread(target=open_when_ready, daemon=True).start()
    print(f'澄明财报助手：{url}')
    print('关闭此窗口或按 Ctrl+C 可停止服务。')
    uvicorn.run(app, host='127.0.0.1', port=port, access_log=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
