# บันทึกสำหรับทำงานต่อใน flybrain-drone

- ระบบบินป่าใช้ FlyVision ใน `px4/forest_pilot.py` ร่วมกับ PX4 หรือ shared pilot ใน fastsim
- ชุด `fly-poc/` มีโมเดล FlyVis pretrained และการทดลองประตู การเปรียบเทียบกับ FlyVision ในสนามป่ายังเปิดอยู่
- ผล PX4 รายเที่ยวอยู่ใน `results/forest/forest_runs.jsonl` และประวัติการทดลองอยู่ใน [EXPERIMENTS.md](../../EXPERIMENTS.md)
- รุ่นอ้างอิง `fix-all2@T600` ถึงทางออก 2/9 ผ่านโดยไม่ชน 0/9 ชนรวม 44 ครั้ง โดย 36 ครั้งใช้ป้ายสถานะ warmup
- [การทดลองเบรกขณะอ้อม](side-guard.md) เปลี่ยนเฉพาะการเดินหน้าในช่วง side เปรียบเทียบกับ fix-all2 โดยใช้โลกและ rep เดียวกัน
