# flybrain-drone

ทดลองว่า "สมองแมลงวัน" (วงจรมองภาพที่รู้จักจาก connectome) ช่วยให้โดรนหลบสิ่งกีดขวางได้ไหม — เริ่มในโปรแกรมจำลอง 2D

โดรนเห็นผ่าน "ตารวม" 120 จุด (รับแค่ความสว่าง ไม่รู้ระยะ) บินในป่าเสา 60 ต้นในสนาม 40×40 ม.
วัดผล: จำนวนช่อง 2×2 ม. ที่บินผ่านก่อนชนครั้งแรก (บินช้า/วนอยู่กับที่ไม่ได้คะแนน), อัตราชน, เวลารอด

## โครงสร้าง

```
flydrone/          โค้ดหลัก
  world.py         สนาม + เสามีลายทาง + raycast ตารวม
  sim.py           วนจำลอง (50 Hz), ตัวชี้วัด
  controllers.py   สมองแต่ละแบบ: no_avoidance, oracle (รู้ระยะจริง), FlyVision (แมลงวัน), Linear (เรียนจากศูนย์)
experiments/       สคริปต์ทดลอง (รันจาก root: python -m experiments.<ชื่อ>)
  h6_finetune.py   H1–H4 + H6: จูนวงจรแมลงวัน vs เรียนจากศูนย์ (งบเท่ากัน) แล้วทดสอบบนโลกที่ไม่เคยเห็น
  bench_compute.py วัดการคำนวณ/เวลาของ "สมอง" ต่อการตัดสินใจ
  diag_signals.py, diag_tau_sweep.py   ตรวจว่าสัญญาณ optic flow ตามระยะจริงแค่ไหน
plots/             plot_summary.py (กราฟรวม), plot_paths.py (เส้นทางบิน)
results/           h6/ compute/ diag/ figures/
EXPERIMENTS.md     บันทึกการทดลองทุกครั้ง
```

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m experiments.h6_finetune        # ~3 นาทีบน M5 (--fast = ไม่กี่วินาที)
.venv/bin/python -m experiments.bench_compute
.venv/bin/python -m plots.plot_summary && .venv/bin/python -m plots.plot_paths
```

## สนามทดสอบป่า (PX4 + Gazebo, `px4/`)

ป่าสร้างด้วยโค้ดทั้งหมด (เรขาคณิต + texture จาก numpy, ไม่มี asset ภายนอก) มีทางเดินยาว 80 ม. แบ่งโซน:
ทางเข้าโล่ง → ลำต้นหนาแน่น (มีต้นยืนในทาง) → ลานโล่ง → กิ่ง/ใบ/พุ่มไม้ระดับตา → โซน texture ต่ำ (ต้นสีเรียบ, ผนังหิน, ก้อนหิน) → ทางออก

```
px4/make_forest.py   สร้างโลก: python -m px4.make_forest {easy|medium|hard} <seed> [--set width=2.8 --set bg_density=0.2 ...]
px4/forest_geom.py   รายการสิ่งกีดขวาง 3D → การชน/ระยะห่าง (ใช้ทั้งฝั่ง host และใน container)
px4/fly_forest.py    บินตามเส้นทางกลางทาง + สมองแมลงวัน (ใน container)
px4/run_forest.sh    sim ใหม่ + บิน 1 เที่ยว (ใน container; 4th arg gui = ดูสดที่ http://localhost:6080/vnc.html กล้องตามโดรน)
px4/forest_configs.json  ชุดพารามิเตอร์ที่ตั้งชื่อ (มีคำอธิบายทุกตัวใน "_doc"), สืบทอดด้วย "base"
px4/forest_params.py     resolve config + กติกาหยุดก่อน (ไม่คืบหน้า 3 ม. ใน 90 s / ชน 20 ครั้ง)
px4/forest_eval.py   ชุดทดลอง → results/forest/forest_runs.jsonl (ทีละเที่ยว พร้อมพารามิเตอร์ครบ), paths_<world>.png
px4/forest_compare.py  ตารางเทียบหลาย run บนโลกเดียวกัน ภายใต้กติกาหยุดเดียวกัน (--md = ตาราง Markdown)
```

การจูน (รันจาก root ของโปรเจกต์):
```bash
.venv/bin/python -m px4.forest_eval --configs lplc2-bold-gate-sv0-v0.8 --param goal_gap=0 --presets medium --workers 3
.venv/bin/python -m px4.forest_compare lplc2-bold-gate-sv0-v0.8 "lplc2-bold-gate-sv0-v0.8+goal_gap=0" --md
```
- `--workers N` = รัน N sim พร้อมกัน (container `flybrain-px4` + `flybrain-px4-w1..` สร้างให้อัตโนมัติ, แยก Gazebo ด้วย GZ_PARTITION, PX4 rootfs แยกต่อ container); 1 sim ≈ 2 core / 1.1 GB → Docker 10 core / 7.7 GB รับได้ ~3
- `--param k=v` แก้พารามิเตอร์ใดก็ได้ใน configs (ชื่อ run = config+k=v อัตโนมัติ), `--speed 2` = เร็วกว่าเวลาจริง
- เที่ยวที่เคยรันแล้ว (preset, seed, run id) ข้ามให้ เว้นแต่ `--redo`

ต้องเปิด container `flybrain-px4` ก่อน (`docker start flybrain-px4`). ดูภาพจากกล้องระดับตา: สร้าง `<world>_preview` ด้วย `preview_world()` แล้วรัน `px4/grab_preview.py`

## สมองแบบแมลงวัน (FlyVision)

- **EMD (Hassenstein–Reichardt)** ระหว่างตาข้างเคียง = สิ่งที่เซลล์ T4/T5 ทำ → optic flow
- **เทียบ flow ซ้าย/ขวา**: ฝั่งที่ภาพไหลเร็ว = ของอยู่ใกล้ → เลี้ยวไปฝั่งที่ไหลช้า
- **ตรวจของพุ่งเข้าหา (คล้าย LPLC2)**: flow ขยายออกด้านหน้า → หักหลบแรง + ชะลอ
- **saccade-and-fixate**: บินตรงวัด → หักเลี้ยวเร็ว (ไม่ดูภาพระหว่างหัก)

ยังไม่ได้ใช้ข้อมูล FlyWire จริง — ตอนนี้เป็นสมการที่วงจรนั้นคำนวณ (H5 คือขั้นต่อไป)
