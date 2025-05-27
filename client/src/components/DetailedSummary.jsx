export default function DetailedSummary({ summary, customCSS }) {
  return (
    <div className={"vertical-widget"} style={customCSS}>
      <h3>Analysis Summary</h3>
      <span>{summary}</span>
    </div>
  );
}
