import { useState } from "react";

export default function HistoryItem({
  patientName = "",
  date = "",
  dominantEmotion = "",
  analysisSummary = "",
  analysisMode = "",
  sessionDuration = "",
}) {
  const temp =
    analysisSummary.length > 50
      ? `${analysisSummary.substring(0, 50)}...`
      : analysisSummary;

  const [summary, setSummary] = useState(temp);
  const [full, setFull] = useState(false);

  function handleSummary() {
    setSummary(analysisSummary);
    setFull(!full);
    if (full) {
      setSummary(temp);
    }
  }

  // Format duration from "00:00:09.447957" to "0 min 9 sec"
  function formatDuration(duration) {
    if (!duration) return "N/A";

    try {
      // Split by : and . to get hours, minutes, seconds, and microseconds
      const parts = duration.split(/[:.]/);

      if (parts.length < 3) return duration; // Return original if not in expected format

      const hours = parseInt(parts[0], 10);
      const minutes = parseInt(parts[1], 10);
      const seconds = parseInt(parts[2], 10);

      // Calculate total minutes including hours
      const totalMinutes = hours * 60 + minutes;

      if (totalMinutes === 0) {
        return `${seconds} sec`;
      } else if (seconds === 0) {
        return `${totalMinutes} min`;
      } else {
        return `${totalMinutes} min ${seconds} sec`;
      }
    } catch (error) {
      console.error("Error formatting duration:", error);
      return duration; // Return original on error
    }
  }

  return (
    <>
      <div
        className="horizontal-widget"
        style={{ width: "90%", padding: "20px" }}
      >
        <div
          className="left-container"
          style={{
            alignItems: "flex-start",
            justifyContent: "flex-start",
            alignSelf: "flex-start",
            margin: 0,
            padding: 0,
          }}
        >
          <h3 className="history-item-heading" style={{ marginTop: 0 }}>
            Patient Email:{" "}
            <span className="history-item-text">{patientName}</span>
          </h3>
          <h3 className="history-item-heading">
            Dominant Emotion:{" "}
            <span className="history-item-text">{dominantEmotion}</span>
          </h3>
          <h3 className="history-item-heading">
            Summary:{" "}
            <span className="history-item-text">
              {summary}
              <span className="show-more" onClick={handleSummary}>
                {!full && summary.length > 50
                  ? " Show More"
                  : summary.length > 50
                  ? " Show Less"
                  : ""}
              </span>
            </span>
          </h3>
        </div>
        <div
          className="right-container"
          style={{
            alignItems: "flex-start",
            justifyContent: "flex-start",
            alignSelf: "flex-start",
            margin: 0,
            padding: 0,
          }}
        >
          <h3 className="history-item-heading" style={{ marginTop: 0 }}>
            Date: <span className="history-item-text">{date}</span>
          </h3>
          <h3 className="history-item-heading">
            Analysis Mode:{" "}
            <span className="history-item-text">{analysisMode}</span>
          </h3>
          <h3 className="history-item-heading">
            Session Duration:{" "}
            <span className="history-item-text">
              {formatDuration(sessionDuration)}
            </span>
          </h3>
        </div>
      </div>
    </>
  );
}
