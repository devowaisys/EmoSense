import Icon from "./Icon";
import cancel from "../assets/icons/cancel.png";
import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function SummaryEditor({
  toggleResetPopup,
  patientInfo,
  summaryID,
  description,
  sessionInfo,
  setSummaryPopupIsVisible,
  emotionAnalysis,
  handleSummarySubmit,
}) {
  const navigate = useNavigate();
  const [editedDescription, setEditedDescription] = useState(description);
  const handleTextAreaChange = (e) => {
    setEditedDescription(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    handleSummarySubmit(editedDescription);
    try {
      const response = await axios.put(
        `http://127.0.0.1:5000/api/update_analysis_text/${summaryID}`,
        {
          analysis_summary: editedDescription,
          patientInfo: patientInfo,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
          timeout: 10000,
        }
      );

      if (response.data.success === true) {
        setSummaryPopupIsVisible(false);
        alert(response.data.message);
        navigate("/home");
      } else {
        alert(response.data.message || "Update failed.");
      }
    } catch (err) {
      if (err.response) {
        alert(err.response.data?.message || "An error occurred on the server.");
      } else if (err.request) {
        alert("Server did not respond. Please try again.");
      } else if (err.code === "ECONNABORTED") {
        alert("Request timed out. Please try again.");
      } else {
        alert(err.message);
      }
    }
  };
  const formatDateTime = (dateTimeString) => {
    const date = new Date(dateTimeString);
    return date.toLocaleString();
  };

  return (
    <div className="popup-outer-container">
      <Icon
        path={cancel}
        customStyle={{ position: "fixed", top: 30 }}
        onClick={toggleResetPopup}
      />

      <h1>Session Summary</h1>
      <form className="popup-inner-form" onSubmit={handleSubmit}>
        <h3 style={{ textDecoration: "underline" }}>Session Details</h3>
        {sessionInfo && (
          <>
            <span className="basic-text">
              Start Time: {formatDateTime(sessionInfo.start_time)}
            </span>
            <br />
            <span className="basic-text">
              End Time: {formatDateTime(sessionInfo.end_time)}
            </span>
            <br />
            <span className="basic-text">
              Duration: {sessionInfo.duration_minutes} minutes
            </span>
          </>
        )}

        <h3 style={{ textDecoration: "underline" }}>Emotional Analysis</h3>
        <span className="basic-text">Dominant Emotion: {emotionAnalysis}</span>

        <h3 style={{ textDecoration: "underline" }}>
          Session Description <span className="basic-text">(Editable)</span>
        </h3>
        <textarea value={editedDescription} onChange={handleTextAreaChange} />

        <div className="button-container">
          <button className={"button"} type="submit">
            Submit
          </button>
        </div>
      </form>
      <span className="notice-text">
        **Note: Submitting will update the auto generated summary and send a
        copy of this report to the client's mentioned email.**
      </span>
    </div>
  );
}
