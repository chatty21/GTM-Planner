Live app: https://chatty21-gtm-planner-streamlit-app-hbi2lj.streamlit.app/

A lightweight decision tool that helps a startup choose which tech YouTubers to sponsor and how to split a fixed budget for the best mix of reach and efficiency.

⸻

What inspired me

I kept seeing creator budgets decided by gut feel—“we’ve always sponsored X” or “that video looked big.” I wanted a portfolio project that felt like a real GTM Analyst workflow: turn public data into benchmarks, transform assumptions into clear plans, and ship it as a simple product non-technical teammates can use. This app is the result.

⸻

What the app does
	•	Benchmark top tech creators on recent reach (30/90-day views), CPM proxies, and ROI proxies.
	•	Plan a budget using three strategies:
	•	Inverse CPM (efficiency): cheapest impressions win more budget
	•	Recent Views (reach): active audience gets priority
	•	Subscribers (fallback upper-funnel)
	•	Risk controls: cap per creator, HHI & top-share metrics, optional diminishing returns penalty.
	•	Sensitivity bands: pessimistic/base/optimistic CPM to show outcome ranges.
	•	Exports: CSV/XLSX plans and a one-pager (Markdown) for sharing.

⸻

How I built it (stack & flow)
	•	App: Streamlit (streamlit_app.py)
	•	Data: public YouTube channel stats (lifetime + recent uploads aggregated to get 30/90-day views).
	•	Modeling:
	•	CPM proxies (90-day when available, lifetime fallback)
	•	Engagement proxy (TotalViews/Subscribers)
	•	Allocation under constraints with optional diminishing returns
	•	Concentration measured via Herfindahl-Hirschman Index (HHI)
	•	Outputs: interactive tables & charts, downloadables (CSV/XLSX/MD).

⸻

Data, assumptions & disclaimers
	•	Uses public stats; no private YouTube Analytics.
	•	CPM/ROI are planning proxies (not sponsor quotes). Great for scenario planning, not contracts.
	•	Optional Category/Region tagging is editable in-app to align with your ICP.
	•	If you have real rate cards or private analytics, plug them in to replace the proxies.

⸻

Try it now

Live app: https://chatty21-gtm-planner-streamlit-app-hbi2lj.streamlit.app/

Upload your CSV in the sidebar or keep a small sample CSV in the repo so the app renders instantly.

⸻

🔧 Using your own data

Minimum columns the app understands (extra columns are fine):
	•	ChannelTitle, Subscribers, TotalViews, VideoCount
	•	Views_30d, Views_90d, Uploads_90d, AvgViewsPerRecent
	•	CPM_90d, CPM_lifetime (proxies), ROI_90d (optional)
	•	SpendUSD (optional), Category, Region (optional)

If a field is missing, the app will create safe defaults; 90-day CPM falls back to lifetime if needed.

⸻

🧮 How to read the plan
	•	Weighted CPM — effective CPM across the whole plan
	•	Projected Impressions — spend / CPM × 1000 (scenario-adjusted)
	•	HHI — concentration risk (0–1; higher = riskier). Keep a lid on top-creator share.
	•	Sensitivity (CPM ±%) — shows range when CPM moves against/for you.

⸻

🛣️ Roadmap
	•	Input real rate cards and compare to proxy CPM
	•	Category/ICP quotas in the allocator
	•	Linear programming optimizer with minimums/caps
	•	Bandit reallocation using live campaign performance
	•	Reach forecasting (ARIMA/Prophet)

⸻

📎 Credits
	•	Built with Streamlit, pandas, numpy, matplotlib
	•	Concept inspired by real startup GTM budgeting pains and creator marketing best practices

⸻

📜 License

MIT — use, modify, and share freely. Feedback and PRs welcome!

⸻

Link again (for convenience)

👉 Live app: https://chatty21-gtm-planner-streamlit-app-hbi2lj.streamlit.app/
