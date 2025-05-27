export default function DetailedInsights({ emotions, customCSS }) {
  return (
    <div className="vertical-widget" style={customCSS}>
      <h3 style={{ marginTop: 0 }}>Detailed Insights</h3>
      <ul>
        {Object.entries(emotions).map(([emotion, { val, emoji }]) => (
          <li key={emotion}>
            {/* In Percentage */}
            <span>{emoji}</span> <strong>{emotion}</strong>: {`${val * 100}%`}
            {/* In Probability */}
            {/* <span>{emoji}</span> <strong>{emotion}</strong>: {val} */}
          </li>
        ))}
      </ul>
    </div>
  );
}
