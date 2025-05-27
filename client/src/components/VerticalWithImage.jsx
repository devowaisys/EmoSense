export default function VerticalWithImage({
  imgPath,
  txt_bold,
  txt_regular,
  customCSS,
  onClick,
}) {
  return (
    <div className="vertical-widget" style={customCSS} onClick={onClick}>
      <img className={"icon"} src={imgPath} alt={"img"} />
      <h3>{`${txt_bold}`}</h3>
      <span style={{ fontWeight: "normal" }}>{txt_regular}</span>
    </div>
  );
}
