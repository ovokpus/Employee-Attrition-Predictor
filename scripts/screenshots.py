"""Capture README screenshots from the running stack: uv run --with playwright python scripts/screenshots.py"""
from playwright.sync_api import sync_playwright

OUT = "docs/screenshots"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1360, "height": 860})
    pg.goto("http://localhost:8080")
    pg.get_by_text("Serving model").wait_for(timeout=30000)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{OUT}/web-high-risk.png", full_page=True)
    pg.get_by_role("button", name="Engaged veteran").click()
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{OUT}/web-low-risk.png", full_page=True)
    pg.goto("http://localhost:5050/#/experiments/1")
    pg.get_by_text("logit-C1.0").first.wait_for(timeout=30000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=f"{OUT}/mlflow-runs.png")
    pg.goto("http://localhost:5050/#/models/employee-attrition")
    pg.get_by_text("champion").first.wait_for(timeout=30000)
    pg.wait_for_timeout(1000)
    pg.screenshot(path=f"{OUT}/mlflow-registry.png")
    pg.goto("http://localhost:8000/docs")
    pg.get_by_text("/predict").first.wait_for()
    pg.screenshot(path=f"{OUT}/api-docs.png", full_page=True)
    b.close()
