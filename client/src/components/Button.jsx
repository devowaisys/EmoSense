export default function Button({
  text,
  onClick,
  width,
  height,
  marginTop,
  imagePath,
  color,
  textColor,
}) {
  return (
    <div
      className={"button"}
      style={{
        width: width,
        height: height,
        marginTop: marginTop,
        backgroundColor: color,
      }}
      onClick={onClick}
    >
      {imagePath ? <img width={30} height={30} src={imagePath} alt="" /> : ""}
      <span className={"btn-text"} style={{ color: textColor }}>
        {text}
      </span>
    </div>
  );
}
