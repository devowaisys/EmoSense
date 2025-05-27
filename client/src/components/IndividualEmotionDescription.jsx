export default function IndividualEmotionDescription({ emotions }) {
  return (
    <div className={"widgets"} style={{ width: "90%" }}>
      {Object.entries(emotions).map(([emotion, { val, emoji }]) => (
        <div className={"vertical-widget individual-vertical-widget"}>
          <span style={{ fontSize: "2rem" }}>{emoji}</span>
          <strong>{emotion}</strong>
          <span>{val}</span>
        </div>
      ))}
    </div>
  );
}
