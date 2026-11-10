from app import create_app
from waitress import serve

app = create_app()

if __name__ == '__main__':
    print('=====================================================')
    print(' Ship Welding Autonomous Platform Running...')
    print(' URL: http://localhost:5000')
    print('=====================================================')

    # [수정 필요] Waitress 프로덕션 WSGI 서버 실행 설정 / 추후 Docker 같은 걸로 실행하는게 포트폴리오에 적합
    # - app: 구동할 Flask WSGI 애플리케이션 객체
    # - host='0.0.0.0': 외부 네트워크 및 타 장비(작업자 PC, 로봇 제어기 등)의 접속 허용
    # - port=5000: 서비스 수신 대기(Listen) 포트 번호
    # - threads=8: 동시 요청 처리를 위한 작업자 스레드 풀 크기 (최대 8개 요청 병렬 처리)
    serve(app, host='0.0.0.0', port=5000, threads=8)
