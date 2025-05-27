import Header from "../components/Header";
import HorizontalText from "../components/HorizontalText";
import waves from "../assets/icons/sound-waves.png";
import HorizontalWithImage from "../components/HorizontalWithImage";
import VerticalWithImage from "../components/VerticalWithImage";
import share from "../assets/icons/share.png";
import language from "../assets/icons/language.png";
import redo from "../assets/icons/redo.png";
import DetailedInsights from "../components/DetailedInsights";
import DetailedSummary from "../components/DetailedSummary";
import { useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AnalysisContext } from "../PreRecordedStore";
import axios from "axios";

export default function PreRecorded() {
  const { analysisResult, patientInfo } = useContext(AnalysisContext);
  const navigate = useNavigate();
  const emotions = {
    Neutral: {
      val: analysisResult?.average_confidence?.neutral || 0,
      emoji: "😐",
    },
    Calm: { val: analysisResult?.average_confidence?.calm || 0, emoji: "😌" },
    Happy: { val: analysisResult?.average_confidence?.happy || 0, emoji: "😊" },
    Sad: { val: analysisResult?.average_confidence?.sad || 0, emoji: "😞" },
    Angry: { val: analysisResult?.average_confidence?.angry || 0, emoji: "😠" },
    Fearful: {
      val: analysisResult?.average_confidence?.fearful || 0,
      emoji: "😰",
    },
    Surprised: {
      val: analysisResult?.average_confidence?.surprised || 0,
      emoji: "😮",
    },
    Disgusted: {
      val: analysisResult?.average_confidence?.disgust || 0,
      emoji: "🤮",
    },
  };

  // Find emotion with highest confidence value
  const dominantEmotion = Object.entries(emotions).reduce(
    (max, [emotion, data]) =>
      data.val > max.val ? { emotion, val: data.val } : max,
    { emotion: "Neutral", val: 0 }
  ).emotion;

  async function sendEmail() {
    console.log(`Sending email for ${patientInfo.name}`);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/email_patient",
        {
          patient_name: patientInfo.name,
          patient_email: patientInfo.email,
          patient_contact: patientInfo.contact,
          analysis_summary: summary,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
          timeout: 10000,
        }
      );

      // Check if the request was successful (status 200 or success flag in response data)
      if (
        response.status === 200 ||
        (response.data && response.data.success === true)
      ) {
        // Display success alert
        alert("Email sent successfully!");
        return true;
      } else {
        // Display failure alert with message if available
        const errorMessage =
          response.data && response.data.message
            ? response.data.message
            : "Failed to send email.";
        alert(`Error: ${errorMessage}`);
        return false;
      }
    } catch (error) {
      // Handle network errors or other exceptions
      const errorMessage =
        error.response && error.response.data && error.response.data.message
          ? error.response.data.message
          : error.message || "An unknown error occurred.";

      // Check if this is a partial success (code 207)
      if (
        error.response &&
        error.response.status === 207 &&
        error.response.data.success === true
      ) {
        alert(
          `Email processing completed, but with warnings: ${
            error.response.data.email_error || errorMessage
          }`
        );
        return true;
      } else {
        alert(`Failed to send email: ${errorMessage}`);
        return false;
      }
    }
  }

  const summary = analysisResult?.description || "No description available.";

  return (
    <>
      <Header />
      <div className={"main-container"}>
        <h1>Pre Recorded Emotion Analysis</h1>
        <div className={"widgets"} style={{ width: "90%" }}>
          <HorizontalText
            txt_bold={"Analysis Mode"}
            txt_regular={"PreRecorded"}
          />
          <HorizontalText
            txt_bold={"Dominant Emotion"}
            txt_regular={dominantEmotion}
            style={{ margin: 0, fontSize: "3rem" }}
          />
          <HorizontalText
            emoji={emotions[dominantEmotion].emoji}
            style={{ margin: 0, fontSize: "3rem" }}
          />
          <HorizontalWithImage
            imgPath={waves}
            txt={
              analysisResult?.duration
                ? `${analysisResult.duration} min`
                : "00:00:00"
            }
            imgCount={2}
          />
          {/* helper component */}
          <VerticalWithImage
            imgPath={language}
            txt_bold={"Language"}
            txt_regular={"English "}
            customCSS={{
              backgroundColor: "#d9d9d9",
            }}
          />
          <DetailedInsights emotions={emotions} />
          <DetailedSummary summary={summary} />
          <VerticalWithImage
            imgPath={share}
            txt_bold={"Share Analysis"}
            onClick={() => sendEmail()}
          />
          <HorizontalWithImage
            imgPath={redo}
            txt={"Run Again"}
            imgCount={1}
            onClick={() => navigate("/home")}
          />
        </div>
      </div>
    </>
  );
}
