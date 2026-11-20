# run.py
import os
import threading
from web import create_web

web = create_web()


def _warmup_embedding_model():
    """서버 기동 시 실제 RAG 검색 경로를 1회 더미 호출하여 싱글톤 임베딩·ChromaDB 사전 적재"""
    try:
        from modules.agent.tools import retriever_tools
        ks = retriever_tools.KnowledgeStore()
        # 실제 질의 시 사용하는 검색 메서드를 1회 호출해 내부 임베딩 싱글톤까지 강제 초기화
        if hasattr(ks, "search"):
            ks.search("warmup")
        elif hasattr(retriever_tools, "search_technical_docs"):
            retriever_tools.search_technical_docs("warmup")
        print("✅ [Warm-up] BAAI/bge-m3 임베딩 및 ChromaDB 사전 적재 완료")
    except Exception as e:
        print(f"⚠️ [Warm-up] 임베딩 사전 로딩 건너뜀: {e}")


if __name__ == '__main__':
    print('=====================================================')
    print(' Ship Welding Autonomous Platform Running...')
    print(' URL: http://localhost:5000')
    print('=====================================================')

    use_waitress = os.getenv("USE_WAITRESS", "false").lower() == "true"

    # Waitress 실행 또는 Flask debug 리로더의 실제 자식 프로세스에서만 1회 워밍업 실행
    if use_waitress or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_warmup_embedding_model, daemon=True).start()

    # .env에 USE_WAITRESS=true를 넣거나 Docker 환경일 때는 Waitress 프로덕션 서버로 구동
    if use_waitress:
        from waitress import serve
        serve(web, host='0.0.0.0', port=5000, threads=8)
    else:
        # 평소 코드 수정하며 테스트할 때 (자동 리로드 지원)
        web.run(host='0.0.0.0', port=5000, debug=True)