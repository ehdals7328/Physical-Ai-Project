from supervisor import run_welding_supervisor

if __name__ == '__main__':
    print("[Agent 단독 테스트 시작]")
    dummy_meta = {'material': 'AH36', 'thickness_mm': 16.0, 'process_type': 'FCAW'}
    res = run_welding_supervisor(dummy_meta, {}, [{'title': '선급규격'}])
    print(f"판정 결과: {res['exec_result']}")
    print(f"지시 사항: {res['work_instruction']['summary_reason']}")
    print("[Agent 단독 테스트 성공!]")
