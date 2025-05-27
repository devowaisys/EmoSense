import pause from "../assets/icons/pause.png";
import resume from "../assets/icons/play-buttton.png";
import stop from "../assets/icons/stop-button.png";

export default function AudioControl({ onPause, onResume, onEnd }) {
  return (
    <div className={"button-container"}>
      <div onClick={onPause}>
        <img src={pause} alt="pause" className={"icon"} />
      </div>
      <div onClick={onResume}>
        <img src={resume} alt="resume" className={"icon"} />
      </div>
      <div onClick={onEnd}>
        <img src={stop} alt="end" className={"icon"} />
      </div>
    </div>
  );
}
