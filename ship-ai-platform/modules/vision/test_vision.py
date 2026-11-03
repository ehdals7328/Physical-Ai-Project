from detector import analyze_welding_frame

if __name__ == '__main__':
    print("[Vision 단독 테스트 시작]")
    res = analyze_welding_frame("dummy_path.jpg")
    print(f"탐지 객체 수: {len(res['detected_objects'])}개")
    print(f"용접선 궤적 좌표 점: {len(res['seam_trajectory'])}개")
    print("[Vision 단독 테스트 성공!]")
