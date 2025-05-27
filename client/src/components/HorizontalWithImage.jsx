export default function HorizontalWithImage({
  imgPath,
  txt,
  imgCount,
  customCSS,
  onClick,
}) {
  return (
    <div className={"horizontal-widget"} style={customCSS} onClick={onClick}>
      {imgCount === 2 ? (
        <>
          <img className={"icon"} src={imgPath} alt={"waves"} />
          <h3 style={{ marginLeft: "5px", marginRight: "5px" }}>{txt}</h3>
          <img className={"icon"} src={imgPath} alt={"waves"} />
        </>
      ) : (
        <>
          <img className={"icon"} src={imgPath} alt={"waves"} />
          <h3 style={{ marginLeft: "5px", marginRight: "5px" }}>{txt}</h3>
        </>
      )}
    </div>
  );
}
