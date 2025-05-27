export default function HorizontalText({
  txt_bold,
  txt_regular,
  emoji,
  style,
  imgPath,
  customCSS,
}) {
  return (
    <div className="horizontal-widget" style={customCSS}>
      {!emoji && !imgPath ? (
        <h3>
          {`${txt_bold}`}{" "}
          <span style={{ fontWeight: "normal" }}>{txt_regular}</span>
        </h3>
      ) : emoji ? (
        <p style={style}>{emoji}</p>
      ) : null}
    </div>
  );
}
