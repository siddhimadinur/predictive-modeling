export default function AboutPage() {
  return (
    <div className="space-y-10">
      <section className="text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 sm:text-4xl">
          About <span className="gradient-text">CalPredict</span>
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-slate-500">
          A machine learning-powered tool for predicting California housing
          prices, built on real census data and modern ensemble methods.
        </p>
      </section>

      {/* Feature cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[
          {
            title: "Accurate Predictions",
            desc: "97.8% R² score using Gradient Boosting — our champion model captures complex non-linear price patterns.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            ),
          },
          {
            title: "California-Specific",
            desc: "Trained exclusively on California housing data, capturing regional nuances from Bay Area tech hubs to Central Valley agriculture.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
            ),
          },
          {
            title: "3 Model Ensemble",
            desc: "Compare predictions across Gradient Boosting, Random Forest, and Ridge Regression to understand model agreement.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            ),
          },
          {
            title: "Confidence Intervals",
            desc: "Tree-based models provide 95% confidence intervals so you understand the range of likely values.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
            ),
          },
          {
            title: "Deep Insights",
            desc: "Explore feature importance, regional patterns, and market drivers to understand what influences pricing.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
            ),
          },
          {
            title: "Production Ready",
            desc: "Deployed on Vercel with Python serverless functions for fast, reliable predictions at scale.",
            icon: (
              <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008zm-3 6h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008z" />
            ),
          },
        ].map((card) => (
          <div key={card.title} className="card-hover p-5">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50">
              <svg className="h-5 w-5 text-brand-600" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                {card.icon}
              </svg>
            </div>
            <h3 className="font-semibold text-slate-900">{card.title}</h3>
            <p className="mt-1.5 text-sm text-slate-500">{card.desc}</p>
          </div>
        ))}
      </div>

      {/* Tech + How to use */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <h2 className="mb-4 text-lg font-bold text-slate-900">
            Technology Stack
          </h2>
          <div className="space-y-3">
            {[
              { name: "Next.js", role: "React framework & frontend" },
              { name: "Vercel", role: "Hosting & Python serverless functions" },
              { name: "scikit-learn", role: "Machine learning models" },
              { name: "Tailwind CSS", role: "Utility-first styling" },
              { name: "Recharts", role: "Data visualization" },
              { name: "pandas / NumPy", role: "Data processing pipeline" },
            ].map((tech) => (
              <div
                key={tech.name}
                className="flex items-center justify-between rounded-lg bg-slate-50 px-4 py-2.5"
              >
                <span className="font-medium text-slate-800">{tech.name}</span>
                <span className="text-sm text-slate-500">{tech.role}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-6">
          <h2 className="mb-4 text-lg font-bold text-slate-900">How to Use</h2>
          <div className="space-y-4">
            {[
              { step: "1", title: "Choose a model", desc: "Select from 3 trained models — Gradient Boosting is recommended." },
              { step: "2", title: "Pick a preset or customize", desc: "Start with a California region preset, then adjust values." },
              { step: "3", title: "Enter property details", desc: "Set income, housing age, rooms, location, and demographics." },
              { step: "4", title: "Get your prediction", desc: "Click Predict to see the estimated value with confidence interval." },
              { step: "5", title: "Explore insights", desc: "Visit Insights and Explorer pages for deeper analysis." },
            ].map((item) => (
              <div key={item.step} className="flex gap-3">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">
                  {item.step}
                </div>
                <div>
                  <p className="font-medium text-slate-800">{item.title}</p>
                  <p className="text-sm text-slate-500">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Dataset info */}
      <div className="card p-6">
        <h2 className="mb-4 text-lg font-bold text-slate-900">
          About the Dataset
        </h2>
        <p className="text-sm leading-relaxed text-slate-600">
          This model is trained on the{" "}
          <strong>California Housing dataset</strong> from scikit-learn, derived
          from the 1990 U.S. Census. It contains 20,640 block-group level
          observations with 8 features including median income, housing age,
          average rooms, population, and geographic coordinates. The target
          variable is the median house value for each block group. While based on
          historical data, the model captures fundamental relationships between
          economic, demographic, and geographic factors that continue to drive
          California real estate pricing.
        </p>
      </div>
    </div>
  );
}
