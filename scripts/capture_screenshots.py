#!/usr/bin/env python3
import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def capture_page(browser, url, name, output_path, wait_ms=5000):
    start = time.time()
    page = await browser.new_page(viewport={"width": 1400, "height": 900})
    await page.goto(url)
    await page.wait_for_timeout(wait_ms)
    await page.screenshot(path=output_path, full_page=False)
    await page.close()
    print(f"  {name}: {time.time()-start:.1f}s -> {output_path}")

async def main():
    base_url = "http://localhost:8501"
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    
    pages = [
        ("Executive Summary", "", "executive_summary.png", 8000),
        ("Dashboard", "dashboard", "dashboard.png", 8000),
        ("Predict", "predict", "predict.png", 5000),
        ("Model Health", "model_health", "model_health.png", 8000),
        ("Business Impact", "business_impact", "business_impact.png", 8000),
    ]
    
    print("Capturing screenshots with extended wait times...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [
            capture_page(browser, f"{base_url}/{path}" if path else base_url, name, str(output_dir/filename), wait)
            for name, path, filename, wait in pages
        ]
        await asyncio.gather(*tasks)
        await browser.close()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
