import asyncio
from mavsdk import System
async def main():
    d = System(); await d.connect(system_address='udpin://0.0.0.0:14540'); print('connecting', flush=True)
    async for s in d.core.connection_state():
        print('conn', s.is_connected, flush=True)
        if s.is_connected: break
    n = 0
    async for h in d.telemetry.health():
        print(h, flush=True); n += 1
        if n > 2: break
asyncio.run(main())
