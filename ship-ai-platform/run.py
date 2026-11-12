# run.py
import os
from web import create_web

web = create_web()

if __name__ == '__main__':
    print('=====================================================')
    print(' Ship Welding Autonomous Platform Running...')
    print(' URL: http://localhost:5000')
    print('=====================================================')

    # .env에 USE_WAITRESS=true를 넣거나 Docker 환경일 때는 Waitress 프로덕션 서버로 구동
    if os.getenv("USE_WAITRESS", "false").lower() == "true":
        from waitress import serve
        serve(web, host='0.0.0.0', port=5000, threads=8)
    else:
        # 평소 코드 수정하며 테스트할 때 (자동 리로드 지원)
        web.run(host='0.0.0.0', port=5000, debug=True)