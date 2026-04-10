export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-10">
      <section className="text-center">
        <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
          About <span className="text-transparent bg-clip-text bg-gradient-to-r from-sunset-500 to-sunset-400">CalPredict</span>
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-gray-500">
          A machine learning-powered tool for predicting California housing
          prices, built on real census data and modern ensemble methods.
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[
          { title: "Accurate Predictions", desc: "97.8% R² score using Gradient Boosting — our champion model captures complex non-linear price patterns." },
          { title: "California-Specific", desc: "Trained exclusively on California housing data, capturing regional nuances from Bay Area tech hubs to Central Valley." },
          { title: "3 Model Ensemble", desc: "Compare predictions across Gradient Boosting, Random Forest, and Ridge Regression." },
          { title: "Confidence Intervals", desc: "Tree-based models provide 95% confidence intervals so you understand the range of likely values." },
          { title: "Deep Insights", desc: "Explore feature importance, regional patterns, and market drivers to understand pricing." },
          { title: "Production Ready", desc: "Deployed on Vercel with Next.js for fast, reliable predictions at scale." },
        ].map((card) => (
          <div key={card.title} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5 hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900">{card.title}</h3>
            <p className="mt-1.5 text-sm text-gray-500">{card.desc}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h2 className="mb-4 text-lg font-bold text-gray-900">Technology Stack</h2>
          <div className="space-y-3">
            {[
              { name: "Next.js", role: "React framework & frontend" },
              { name: "Vercel", role: "Hosting & deployment" },
              { name: "scikit-learn", role: "Machine learning models" },
              { name: "Tailwind CSS", role: "Utility-first styling" },
              { name: "Recharts", role: "Data visualization" },
              { name: "pandas / NumPy", role: "Data processing pipeline" },
            ].map((tech) => (
              <div key={tech.name} className="flex items-center justify-between rounded-lg bg-gray-50 px-4 py-2.5">
                <span className="font-medium text-gray-800">{tech.name}</span>
                <span className="text-sm text-gray-500">{tech.role}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h2 className="mb-4 text-lg font-bold text-gray-900">How to Use</h2>
          <div className="space-y-4">
            {[
              { step: "1", title: "Pick a city", desc: "Select from 20 California cities — area data is auto-filled." },
              { step: "2", title: "Adjust your property", desc: "Use the sliders to set rooms, bedrooms, and household size." },
              { step: "3", title: "See your estimate", desc: "Get both 1990 census and inflation-adjusted 2024 values instantly." },
              { step: "4", title: "Explore insights", desc: "Visit Insights for model performance and feature importance." },
              { step: "5", title: "Compare cities", desc: "Use the Explorer to compare prices across California." },
            ].map((item) => (
              <div key={item.step} className="flex gap-3">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sunset-500 text-xs font-bold text-white">{item.step}</div>
                <div>
                  <p className="font-medium text-gray-800">{item.title}</p>
                  <p className="text-sm text-gray-500">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
        <h2 className="mb-4 text-lg font-bold text-gray-900">About the Dataset</h2>
        <p className="text-sm leading-relaxed text-gray-600">
          This model is trained on the <strong>California Housing dataset</strong> from scikit-learn, derived
          from the 1990 U.S. Census. It contains 20,640 block-group level observations with 8 features
          including median income, housing age, average rooms, population, and geographic coordinates.
          The target variable is the median house value for each block group. While based on historical data,
          the model captures fundamental relationships between economic, demographic, and geographic factors
          that continue to drive California real estate pricing.
        </p>
      </div>
    </div>
  );
}
