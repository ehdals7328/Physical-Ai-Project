from pathlib import Path
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Mm, Pt, RGBColor


def set_cell_bg(cell, hex_color: str):
  shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
  cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=38, bottom=38, left=90, right=90):
  tcPr = cell._tc.get_or_add_tcPr()
  tcMar = OxmlElement("w:tcMar")
  for m, val in [
      ("top", top),
      ("bottom", bottom),
      ("left", left),
      ("right", right),
  ]:
    node = OxmlElement(f"w:{m}")
    node.set(qn("w:w"), str(val))
    node.set(qn("w:type"), "dxa")
    tcMar.append(node)
  tcPr.append(tcMar)


def set_table_borders(table):
  tblPr = table._tbl.tblPr
  borders = parse_xml(f"""
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="333333"/>
            <w:bottom w:val="single" w:sz="6" w:space="0" w:color="333333"/>
            <w:left w:val="single" w:sz="6" w:space="0" w:color="333333"/>
            <w:right w:val="single" w:sz="6" w:space="0" w:color="333333"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="333333"/>
            <w:insideV w:val="single" w:sz="4" w:space="0" w:color="333333"/>
        </w:tblBorders>
    """)
  tblPr.append(borders)


def style_run(run, size=8.0, bold=False, color=(17, 17, 17)):
  run.font.name = "맑은 고딕"
  run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
  run.font.size = Pt(size)
  run.bold = bold
  run.font.color.rgb = RGBColor(*color)


def fill_cell(cell, text, is_header=False, bold=False):
  cell.text = ""
  cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
  set_cell_margins(cell)
  if is_header:
    set_cell_bg(cell, "E8EEF5")
  p = cell.paragraphs[0]
  p.paragraph_format.space_before = Pt(0)
  p.paragraph_format.space_after = Pt(0)
  run = p.add_run(text)
  style_run(run, size=8.5, bold=(is_header or bold))


def add_section_banner(table, title_text: str):
  row = table.add_row()
  merged = row.cells[0].merge(row.cells[3])
  merged.text = ""
  merged.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
  set_cell_bg(merged, "1F4E79")
  set_cell_margins(merged, top=70, bottom=70, left=110, right=110)
  p = merged.paragraphs[0]
  p.paragraph_format.space_before = Pt(0)
  p.paragraph_format.space_after = Pt(0)
  run = p.add_run(title_text)
  style_run(run, size=9.5, bold=True, color=(255, 255, 255))


def add_kv_row(table, k1, v1, k2, v2, bold_v1=False, bold_v2=False):
  row = table.add_row()
  widths = [Mm(34), Mm(59), Mm(34), Mm(59)]
  for idx, w in enumerate(widths):
    row.cells[idx].width = w
  fill_cell(row.cells[0], k1, is_header=True)
  fill_cell(row.cells[1], v1, bold=bold_v1)
  fill_cell(row.cells[2], k2, is_header=True)
  fill_cell(row.cells[3], v2, bold=bold_v2)


def add_span3_row(table, k1, v1):
  row = table.add_row()
  row.cells[0].width = Mm(34)
  merged = row.cells[1].merge(row.cells[3])
  merged.width = Mm(152)
  fill_cell(row.cells[0], k1, is_header=True)
  fill_cell(merged, v1)


def build_template():
  out_path = Path("./generated_docs/templates/work_order_template.docx")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  doc = Document()
  sec = doc.sections[0]
  sec.page_width = Mm(210)
  sec.page_height = Mm(297)
  sec.top_margin = Mm(9)
  sec.bottom_margin = Mm(9)
  sec.left_margin = Mm(11)
  sec.right_margin = Mm(11)

  title_p = doc.add_paragraph()
  title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
  title_p.paragraph_format.space_after = Pt(8)
  t_run = title_p.add_run(
      "선 박 블 록 용 접 작 업 지 시 서 (WELDING WORK ORDER)"
  )
  style_run(t_run, size=14.5, bold=True)

  # 상단 결재/문서 메타 테이블
  meta_tbl = doc.add_table(rows=0, cols=4)
  meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
  meta_tbl.autofit = False
  set_table_borders(meta_tbl)

  add_kv_row(
      meta_tbl,
      "문서번호",
      "{{ work_order_number }}",
      "개정 / 상태",
      "{{ revision_no }} / {{ approval_status }}",
      True,
      True,
  )
  add_kv_row(
      meta_tbl,
      "발행일자",
      "{{ created_date }}",
      "결재 (작성/검토/승인)",
      (
          "작성: AI Agent | 검토: {{ reviewed_by }} | 승인: {{ approved_by"
          " }}"
      ),
  )
  add_kv_row(
      meta_tbl,
      "수행구분",
      "{{ execution_type }}",
      "협력사명",
      "{{ partner_name }}",
      True,
      False,
  )

  sp = doc.add_paragraph()
  sp.paragraph_format.space_before = Pt(2)
  sp.paragraph_format.space_after = Pt(2)

  # 본문 5단 섹션 테이블
  main_tbl = doc.add_table(rows=0, cols=4)
  main_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
  main_tbl.autofit = False
  set_table_borders(main_tbl)

  # 1. 대상 블록 및 공정 위치
  add_section_banner(main_tbl, "■ 1. 대상 블록 및 공정 위치 정보")
  add_kv_row(
      main_tbl,
      "대상 호선 (SHIP_CODE)",
      "{{ ship_code }}",
      "대상 블록 (BLOCK_CODE)",
      "{{ block_id }}",
      False,
      True,
  )
  add_kv_row(
      main_tbl,
      "용접 부재 / 블록명",
      "{{ block_name }}",
      "블록 중량 (WEIGHT_TON)",
      "{{ weight_ton }} 톤",
  )
  add_kv_row(
      main_tbl,
      "작업 야드 (BRANCH)",
      "{{ branch_name }}",
      "작업 구역 (WORK_ZONE)",
      "{{ work_zone }}",
  )

  # 2. 투입 인력·자재·장비
  add_section_banner(main_tbl, "■ 2. 투입 인력 · 자재 · 장비 배정 현황")
  add_kv_row(
      main_tbl,
      "담당 작업자 / 사번",
      "{{ worker_name }} ({{ worker_id }})",
      "선급 자격 / 인증공정",
      "{{ cert_code }} ({{ weld_process }})",
      True,
      False,
  )
  add_kv_row(
      main_tbl,
      "배정 강재 Lot",
      "{{ allocated_lot_no }}",
      "용접 와이어 / 선점량",
      "{{ wire_spec_text }} / {{ reserved_wire_kg }} kg",
  )
  add_kv_row(
      main_tbl,
      "배차 장비 / 차량",
      "{{ assigned_vehicle_code }}",
      "외주 발주(PO) 번호",
      "{{ b2b_po_number }}",
  )

  # 3. WPS 용접 시공 조건 및 현장 비전 안전 인터록 (좌표/길이 제거, WPS/품목명 반영)
  add_section_banner(
      main_tbl, "■ 3. WPS 용접 시공 조건 및 현장 비전 안전 인터록"
  )
  add_kv_row(
      main_tbl,
      "적용 절차서 (WPS NO)",
      "{{ wps_code }}",
      "모재 강종 / 두께",
      "{{ material }} / {{ thickness }} mm",
      True,
      True,
  )
  add_kv_row(
      main_tbl,
      "용접 공정 / 이음형상",
      "{{ process }} / {{ joint_type }} ({{ weld_position }})",
      "용가재 규격 / 품목",
      "{{ wire_spec_text }}",
      True,
      False,
  )
  add_kv_row(
      main_tbl,
      "표준 전류 / 전압",
      "{{ current_amp }} A / {{ voltage_v }} V",
      "이송 속도 / 보호가스",
      "{{ travel_speed }} cm/min / {{ shielding_gas }}",
  )
  add_kv_row(
      main_tbl,
      "예열 및 층간온도",
      "{{ preheat_temp }}",
      "선급 품질검사 (NDT)",
      "{{ ndt_requirement }}",
  )
  add_span3_row(main_tbl, "용접재료 건조·관리", "{{ consumable_spec }}")
  add_kv_row(
      main_tbl,
      "YOLOv11 비전 탐지 상태",
      "{{ vision_status_text }}",
      "작업 구역 안전 클리어런스",
      "{{ safety_clearance_text }}",
      False,
      True,
  )

  # 4. 실시간 현장 기상 환경 및 필수 안전 준수사항
  add_section_banner(
      main_tbl,
      "■ 4. 실시간 현장 기상 환경 및 필수 안전 준수사항 (Safety Interlock)",
  )
  add_span3_row(
      main_tbl,
      "실시간 기상 스냅샷",
      (
          "관측 야드: {{ location }} | 기온: {{ temp }}℃ | 풍속: {{ wind }}m/s"
          " | 강수량: {{ rain }}mm/h | 안전 판정: {{ safety_status }}"
      ),
  )
  add_span3_row(main_tbl, "필수 안전 체크리스트", "{{ safety_checklist }}")

  # 5. 현장 작업 완료 확인 및 근거 규격
  add_section_banner(main_tbl, "■ 5. 현장 작업 완료 확인 및 근거 규격")
  add_kv_row(
      main_tbl,
      "작업자 확인",
      "{{ worker_name }} (인)",
      "현장 반장 확인",
      "____________________ (인)",
  )
  add_span3_row(main_tbl, "참조 규격 (RAG 근거)", "{{ source_refs }}")

  doc.save(str(out_path))
  print(f"✅ 표준 작업지시서 템플릿 갱신 완료: {out_path.resolve()}")


if __name__ == "__main__":
  build_template()