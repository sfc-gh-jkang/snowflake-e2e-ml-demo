#!/usr/bin/env python3
import asyncio
import sys
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def capture_page(browser, url, name, output_path, wait_ms=4000, actions=None):
    start = time.time()
    page = await browser.new_page(viewport={"width": 1400, "height": 900})
    await page.goto(url)
    await page.wait_for_timeout(wait_ms)
    
    if actions:
        for action in actions:
            if action.get('click'):
                try:
                    await page.click(action['click'])
                    await page.wait_for_timeout(action.get('wait', 1500))
                except Exception as e:
                    print(f"  Warning: Could not click '{action['click']}': {e}")
            if action.get('scroll'):
                await page.evaluate(f"window.scrollBy(0, {action['scroll']})")
                await page.wait_for_timeout(500)
    
    await page.screenshot(path=output_path, full_page=False)
    await page.close()
    print(f"  {name}: {time.time()-start:.1f}s -> {output_path}")
    return time.time() - start

async def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./docs")
    output_dir.mkdir(exist_ok=True)
    
    pages = [
        ("Executive Summary", f"{base_url}", "executive_summary.png", None),
        ("Dashboard", f"{base_url}/dashboard", "dashboard.png", [
            {'click': 'text=What is Churn Prediction?', 'wait': 1000}
        ]),
        ("Prediction", f"{base_url}/predict", "predict.png", [
            {'click': '[data-testid="stSelectbox"]', 'wait': 500},
            {'click': 'text=High Risk', 'wait': 1500}
        ]),
        ("Model Health", f"{base_url}/model_health", "model_health.png", None),
        ("Business Impact", f"{base_url}/business_impact", "business_impact.png", None),
    ]
    
    print(f"Capturing {len(pages)} screenshots in PARALLEL from {base_url}...")
    start_total = time.time()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [
            capture_page(browser, url, name, str(output_dir/filename), 5000, actions)
            for name, url, filename, actions in pages
        ]
        await asyncio.gather(*tasks)
        await browser.close()
    
    print(f"\nTotal time: {time.time()-start_total:.1f}s (parallel)")
    print(f"Screenshots saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
