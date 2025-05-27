// import { useState, useRef, useEffect, useContext, useCallback } from "react";
// import { useParams } from "react-router-dom";
// import { io } from "socket.io-client";
// import Header from "../components/Header";
// import Button from "../components/Button";
// import mic from "../assets/icons/mic-rec.png";
// import AudioControl from "../components/AudioControl";
// import IndividualEmotionDescription from "../components/IndividualEmotionDescription";
// import { UserContext } from "../UserStore";
// import SummaryEditor from "../components/SummaryEditor";

// export default function Realtime() {
//   const [summaryPopupIsVisible, setSummaryPopupIsVisible] = useState(false);
//   const [summary, setSummary] = useState({
//     id: 0,
//     description: "",
//     sessionInfo: null,
//     emotionAnalysis: null,
//   });
//   console.log(summary);
//   const [elapsed, setElapsed] = useState(0);
//   const [isPaused, setIsPaused] = useState(false);
//   const [isConnected, setIsConnected] = useState(false);
//   const [sessionId, setSessionId] = useState(null);
//   const [dominantEmotion, setDominantEmotion] = useState("Calm");

//   const { user } = useContext(UserContext);
//   const timerRef = useRef(null);
//   const socketRef = useRef(null);
//   const mediaRecorderRef = useRef(null);
//   const streamRef = useRef(null);
//   const { email, fullname, contact } = useParams();

//   const [emotions, setEmotions] = useState({
//     Neutral: { val: 0, emoji: "😐" },
//     Calm: { val: 0, emoji: "😌" },
//     Happy: { val: 0, emoji: "😊" },
//     Sad: { val: 0, emoji: "😞" },
//     Angry: { val: 0, emoji: "😠" },
//     Fearful: { val: 0, emoji: "😰" },
//     Surprised: { val: 0, emoji: "😮" },
//     Disgusted: { val: 0, emoji: "🤮" },
//   });

//   const patientInfo = {
//     email: decodeURIComponent(email),
//     fullname: decodeURIComponent(fullname),
//     contact: decodeURIComponent(contact),
//   };

//   function toggleResetPopup() {
//     setSummaryPopupIsVisible(false);
//   }

//   const handleSummarySubmit = (updatedSummary) => {
//     setSummary((prev) => ({
//       ...prev,
//       description: updatedSummary,
//     }));
//     // setSummaryPopupIsVisible(false);
//   };

//   const capitalizeFirstLetter = useCallback((string) => {
//     return string.charAt(0).toUpperCase() + string.slice(1);
//   }, []);

//   const updateEmotionsFromResult = useCallback(
//     (result) => {
//       if (!result || !result.confidence_scores) return;

//       setEmotions((prev) => ({
//         Neutral: {
//           ...prev.Neutral,
//           val: `${Math.round(result.confidence_scores.neutral * 100)}%`,
//         },
//         Calm: {
//           ...prev.Calm,
//           val: `${Math.round(result.confidence_scores.calm * 100)}%`,
//         },
//         Happy: {
//           ...prev.Happy,
//           val: `${Math.round(result.confidence_scores.happy * 100)}%`,
//         },
//         Sad: {
//           ...prev.Sad,
//           val: `${Math.round(result.confidence_scores.sad * 100)}%`,
//         },
//         Angry: {
//           ...prev.Angry,
//           val: `${Math.round(result.confidence_scores.angry * 100)}%`,
//         },
//         Fearful: {
//           ...prev.Fearful,
//           val: `${Math.round(result.confidence_scores.fearful * 100)}%`,
//         },
//         Surprised: {
//           ...prev.Surprised,
//           val: `${Math.round(result.confidence_scores.surprised * 100)}%`,
//         },
//         Disgusted: {
//           ...prev.Disgusted,
//           val: `${Math.round(result.confidence_scores.disgust * 100)}%`,
//         },
//       }));

//       setDominantEmotion(capitalizeFirstLetter(result.predicted_emotion));
//     },
//     [capitalizeFirstLetter]
//   );

//   useEffect(() => {
//     const jwt = "your_jwt_token";
//     socketRef.current = io("http://localhost:5000", {
//       auth: { token: jwt },
//     });

//     socketRef.current.on("connect", () => {
//       console.log("Connected");
//       setIsConnected(true);
//     });

//     socketRef.current.on("connect_error", (error) => {
//       console.log("Connection Error:", error.message);
//       setIsConnected(false);
//     });

//     socketRef.current.on("analysis_started", (data) => {
//       console.log("Session started:", data.message);
//     });

//     socketRef.current.on("analysis_result", (data) => {
//       console.log("Analysis result:", data.result);
//       updateEmotionsFromResult(data.result);
//     });

//     socketRef.current.on("analysis_ended", (data) => {
//       console.log("Session ended:", data);
//       console.log(data.summary);
//       if (data.status === "success") {
//         setSummary({
//           id: data.summary.save_result.analysis_id,
//           description: data.summary.summary.summary_description,
//           sessionInfo: data.summary.summary.session_info,
//           emotionAnalysis: Object.keys(
//             data.summary.summary.emotion_analysis.dominant_emotions
//           )[0],
//         });
//         setSummaryPopupIsVisible(true);
//       }
//     });

//     socketRef.current.on("error", (error) => {
//       console.log("Error:", error.message);
//     });

//     socketRef.current.on("disconnect", () => {
//       console.log("Disconnected");
//       setIsConnected(false);
//     });

//     return () => {
//       if (socketRef.current) {
//         socketRef.current.disconnect();
//       }
//     };
//   }, [updateEmotionsFromResult]);

//   const formatTime = (seconds) => {
//     const minutes = Math.floor(seconds / 60)
//       .toString()
//       .padStart(2, "0");
//     const secs = (seconds % 60).toString().padStart(2, "0");
//     return `${minutes}:${secs}`;
//   };

//   const startSession = useCallback(() => {
//     const newSessionId = "session_" + Date.now();
//     setSessionId(newSessionId);

//     socketRef.current.emit("start_analysis", {
//       session_id: newSessionId,
//       patient_fullname: patientInfo.fullname,
//       patient_email: patientInfo.email,
//       patient_contact: patientInfo.contact,
//       therapist_id: user.id,
//     });

//     return newSessionId;
//   }, [patientInfo.fullname, patientInfo.email, patientInfo.contact, user.id]);

//   const handleStart = async () => {
//     try {
//       const newSessionId = startSession();

//       const stream = await navigator.mediaDevices.getUserMedia({
//         audio: {
//           channelCount: 1,
//           sampleRate: 44100,
//         },
//       });

//       streamRef.current = stream;
//       mediaRecorderRef.current = new MediaRecorder(stream, {
//         mimeType: "audio/webm;codecs=opus",
//         bitsPerSecond: 128000,
//       });

//       mediaRecorderRef.current.ondataavailable = async (event) => {
//         if (event.data.size > 0) {
//           try {
//             const arrayBuffer = await event.data.arrayBuffer();
//             const uint8Array = new Uint8Array(arrayBuffer);
//             const base64Data = btoa(
//               uint8Array.reduce(
//                 (data, byte) => data + String.fromCharCode(byte),
//                 ""
//               )
//             );

//             socketRef.current.emit("audio_chunk", {
//               session_id: newSessionId,
//               audio_data: base64Data,
//             });
//           } catch (error) {
//             console.error("Error processing audio chunk:", error);
//           }
//         }
//       };

//       mediaRecorderRef.current.start(1000);

//       timerRef.current = setInterval(() => {
//         setElapsed((prevElapsed) => prevElapsed + 1);
//       }, 1000);

//       setIsPaused(false);
//     } catch (error) {
//       console.error("Error starting recording:", error);
//     }
//   };

//   const handlePause = () => {
//     if (
//       mediaRecorderRef.current &&
//       mediaRecorderRef.current.state === "recording"
//     ) {
//       mediaRecorderRef.current.pause();
//     }
//     if (timerRef.current !== null) {
//       clearInterval(timerRef.current);
//       timerRef.current = null;
//       setIsPaused(true);
//     }
//   };

//   const handleResume = () => {
//     if (
//       mediaRecorderRef.current &&
//       mediaRecorderRef.current.state === "paused"
//     ) {
//       mediaRecorderRef.current.resume();
//     }
//     if (timerRef.current === null && isPaused) {
//       timerRef.current = setInterval(() => {
//         setElapsed((prevElapsed) => prevElapsed + 1);
//       }, 1000);
//       setIsPaused(false);
//     }
//   };

//   const handleEnd = () => {
//     if (mediaRecorderRef.current) {
//       mediaRecorderRef.current.stop();
//       streamRef.current?.getTracks().forEach((track) => track.stop());
//     }

//     if (socketRef.current && sessionId) {
//       socketRef.current.emit("end_analysis", { session_id: sessionId });
//     }

//     if (timerRef.current !== null) {
//       clearInterval(timerRef.current);
//       timerRef.current = null;
//       setElapsed(0);
//       setIsPaused(false);
//     }

//     setSessionId(null);
//   };

//   return (
//     <>
//       <Header />
//       {summaryPopupIsVisible && (
//         <SummaryEditor
//           toggleResetPopup={toggleResetPopup}
//           patientInfo={patientInfo}
//           summaryID={summary.id}
//           description={summary.description}
//           sessionInfo={summary.sessionInfo}
//           setSummaryPopupIsVisible={setSummaryPopupIsVisible}
//           emotionAnalysis={summary.emotionAnalysis}
//           handleSummarySubmit={handleSummarySubmit}
//         />
//       )}

//       <div className="main-container">
//         <h1>Realtime Emotion Analysis</h1>
//         <h3>Recording Time Elapsed: {formatTime(elapsed)}</h3>
//         {!isConnected && <span>Not connected to server</span>}
//         {isConnected && timerRef.current === null && !isPaused && (
//           <Button
//             imagePath={mic}
//             text="Start Recording"
//             onClick={handleStart}
//           />
//         )}
//         {(timerRef.current !== null || isPaused) && (
//           <AudioControl
//             onPause={handlePause}
//             onResume={handleResume}
//             onEnd={handleEnd}
//             isPaused={isPaused}
//           />
//         )}
//         <h3>Dominant Emotion: {dominantEmotion}</h3>
//         <IndividualEmotionDescription emotions={emotions} />
//       </div>
//     </>
//   );
// }

////////////////////////////////////////////////////////////////////////////////////////////////
// import { useState, useRef, useEffect, useContext, useCallback } from "react";
// import { useParams } from "react-router-dom";
// import { io } from "socket.io-client";
// import Header from "../components/Header";
// import Button from "../components/Button";
// import mic from "../assets/icons/mic-rec.png";
// import AudioControl from "../components/AudioControl";
// import IndividualEmotionDescription from "../components/IndividualEmotionDescription";
// import { UserContext } from "../UserStore";
// import SummaryEditor from "../components/SummaryEditor";

// export default function Realtime() {
//   const [summaryPopupIsVisible, setSummaryPopupIsVisible] = useState(false);
//   const [summary, setSummary] = useState({
//     id: 0,
//     description: "",
//     sessionInfo: null,
//     emotionAnalysis: null,
//   });
//   const [elapsed, setElapsed] = useState(0);
//   const [isPaused, setIsPaused] = useState(false);
//   const [isConnected, setIsConnected] = useState(false);
//   const [sessionId, setSessionId] = useState(null);
//   const [dominantEmotion, setDominantEmotion] = useState("Calm");

//   const { user } = useContext(UserContext);
//   const timerRef = useRef(null);
//   const socketRef = useRef(null);
//   const mediaRecorderRef = useRef(null);
//   const streamRef = useRef(null);
//   const { email, fullname, contact } = useParams();

//   const [emotions, setEmotions] = useState({
//     Neutral: { val: 0, emoji: "😐" },
//     Calm: { val: 0, emoji: "😌" },
//     Happy: { val: 0, emoji: "😊" },
//     Sad: { val: 0, emoji: "😞" },
//     Angry: { val: 0, emoji: "😠" },
//     Fearful: { val: 0, emoji: "😰" },
//     Surprised: { val: 0, emoji: "😮" },
//     Disgusted: { val: 0, emoji: "🤮" },
//   });

//   const patientInfo = {
//     email: decodeURIComponent(email),
//     fullname: decodeURIComponent(fullname),
//     contact: decodeURIComponent(contact),
//   };

//   function toggleResetPopup() {
//     setSummaryPopupIsVisible(false);
//   }

//   const handleSummarySubmit = (updatedSummary) => {
//     setSummary((prev) => ({
//       ...prev,
//       description: updatedSummary,
//     }));
//   };

//   const capitalizeFirstLetter = useCallback((string) => {
//     return string.charAt(0).toUpperCase() + string.slice(1);
//   }, []);

//   const updateEmotionsFromResult = useCallback(
//     (result) => {
//       if (!result || !result.confidence_scores) return;

//       // Ensure we're working with numbers, not strings
//       setEmotions((prev) => ({
//         Neutral: {
//           ...prev.Neutral,
//           val: Math.round(result.confidence_scores.neutral * 100) + "%",
//         },
//         Calm: {
//           ...prev.Calm,
//           val: Math.round(result.confidence_scores.calm * 100) + "%",
//         },
//         Happy: {
//           ...prev.Happy,
//           val: Math.round(result.confidence_scores.happy * 100) + "%",
//         },
//         Sad: {
//           ...prev.Sad,
//           val: Math.round(result.confidence_scores.sad * 100) + "%",
//         },
//         Angry: {
//           ...prev.Angry,
//           val: Math.round(result.confidence_scores.angry * 100) + "%",
//         },
//         Fearful: {
//           ...prev.Fearful,
//           val: Math.round(result.confidence_scores.fearful * 100) + "%",
//         },
//         Surprised: {
//           ...prev.Surprised,
//           val: Math.round(result.confidence_scores.surprised * 100) + "%",
//         },
//         Disgusted: {
//           ...prev.Disgusted,
//           val: Math.round(result.confidence_scores.disgust * 100) + "%",
//         },
//       }));

//       // Log the predicted emotion for debugging
//       console.log(
//         `Received emotion: ${result.predicted_emotion} with scores:`,
//         result.confidence_scores
//       );

//       setDominantEmotion(capitalizeFirstLetter(result.predicted_emotion));
//     },
//     [capitalizeFirstLetter]
//   );

//   useEffect(() => {
//     const jwt = "your_jwt_token";
//     socketRef.current = io("http://localhost:5000", {
//       auth: { token: jwt },
//     });

//     socketRef.current.on("connect", () => {
//       console.log("Connected");
//       setIsConnected(true);
//     });

//     socketRef.current.on("connect_error", (error) => {
//       console.log("Connection Error:", error.message);
//       setIsConnected(false);
//     });

//     socketRef.current.on("analysis_started", (data) => {
//       console.log("Session started:", data.message);
//     });

//     socketRef.current.on("analysis_result", (data) => {
//       console.log("Analysis result:", data.result);
//       updateEmotionsFromResult(data.result);
//     });

//     socketRef.current.on("analysis_ended", (data) => {
//       console.log("Session ended:", data);
//       if (data.status === "success") {
//         setSummary({
//           id: data.summary.save_result.analysis_id,
//           description: data.summary.summary.summary_description,
//           sessionInfo: data.summary.summary.session_info,
//           emotionAnalysis: Object.keys(
//             data.summary.summary.emotion_analysis.dominant_emotions
//           )[0],
//         });
//         setSummaryPopupIsVisible(true);
//       }
//     });

//     socketRef.current.on("error", (error) => {
//       console.log("Error:", error.message);
//     });

//     socketRef.current.on("disconnect", () => {
//       console.log("Disconnected");
//       setIsConnected(false);
//     });

//     return () => {
//       if (socketRef.current) {
//         socketRef.current.disconnect();
//       }
//     };
//   }, [updateEmotionsFromResult]);

//   const formatTime = (seconds) => {
//     const minutes = Math.floor(seconds / 60)
//       .toString()
//       .padStart(2, "0");
//     const secs = (seconds % 60).toString().padStart(2, "0");
//     return `${minutes}:${secs}`;
//   };

//   const startSession = useCallback(() => {
//     const newSessionId = "session_" + Date.now();
//     setSessionId(newSessionId);

//     socketRef.current.emit("start_analysis", {
//       session_id: newSessionId,
//       patient_fullname: patientInfo.fullname,
//       patient_email: patientInfo.email,
//       patient_contact: patientInfo.contact,
//       therapist_id: user.id,
//     });

//     return newSessionId;
//   }, [patientInfo.fullname, patientInfo.email, patientInfo.contact, user.id]);

//   const handleStart = async () => {
//     try {
//       const newSessionId = startSession();

//       // Request audio with proper constraints
//       const stream = await navigator.mediaDevices.getUserMedia({
//         audio: {
//           channelCount: 1,
//           sampleRate: 16000, // Use 16kHz to match common ML models
//           echoCancellation: true,
//           noiseSuppression: true,
//         },
//       });

//       streamRef.current = stream;

//       // Use raw PCM encoding for better compatibility with ML models
//       const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
//         sampleRate: 16000,
//       });
//       const source = audioCtx.createMediaStreamSource(stream);
//       const processor = audioCtx.createScriptProcessor(4096, 1, 1);

//       source.connect(processor);
//       processor.connect(audioCtx.destination);

//       processor.onaudioprocess = (e) => {
//         if (socketRef.current && socketRef.current.connected) {
//           const inputData = e.inputBuffer.getChannelData(0);

//           // Convert to 16-bit PCM for better compatibility
//           const pcmData = new Int16Array(inputData.length);
//           for (let i = 0; i < inputData.length; i++) {
//             pcmData[i] = inputData[i] * 32767;
//           }

//           // Send as base64 encoded PCM data
//           const base64Data = btoa(
//             String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer))
//           );

//           socketRef.current.emit("audio_chunk", {
//             session_id: newSessionId,
//             audio_data: base64Data,
//             sample_rate: 16000,
//             format: "pcm_16bit",
//           });
//         }
//       };

//       // Store the processor for cleanup
//       mediaRecorderRef.current = { processor, source, audioCtx };

//       timerRef.current = setInterval(() => {
//         setElapsed((prevElapsed) => prevElapsed + 1);
//       }, 1000);

//       setIsPaused(false);
//     } catch (error) {
//       console.error("Error starting recording:", error);
//     }
//   };

//   const handlePause = () => {
//     if (mediaRecorderRef.current) {
//       // Disconnect the processor to pause audio processing
//       mediaRecorderRef.current.source.disconnect();
//     }

//     if (timerRef.current !== null) {
//       clearInterval(timerRef.current);
//       timerRef.current = null;
//       setIsPaused(true);
//     }
//   };

//   const handleResume = () => {
//     if (mediaRecorderRef.current) {
//       // Reconnect the processor to resume audio processing
//       mediaRecorderRef.current.source.connect(
//         mediaRecorderRef.current.processor
//       );
//       mediaRecorderRef.current.processor.connect(
//         mediaRecorderRef.current.audioCtx.destination
//       );
//     }

//     if (timerRef.current === null && isPaused) {
//       timerRef.current = setInterval(() => {
//         setElapsed((prevElapsed) => prevElapsed + 1);
//       }, 1000);
//       setIsPaused(false);
//     }
//   };

//   const handleEnd = () => {
//     if (mediaRecorderRef.current) {
//       // Clean up audio processing
//       mediaRecorderRef.current.source.disconnect();
//       mediaRecorderRef.current.processor.disconnect();
//       mediaRecorderRef.current.audioCtx.close().catch(console.error);
//     }

//     if (streamRef.current) {
//       streamRef.current.getTracks().forEach((track) => track.stop());
//     }

//     if (socketRef.current && sessionId) {
//       socketRef.current.emit("end_analysis", { session_id: sessionId });
//     }

//     if (timerRef.current !== null) {
//       clearInterval(timerRef.current);
//       timerRef.current = null;
//       setElapsed(0);
//       setIsPaused(false);
//     }

//     setSessionId(null);
//   };

//   return (
//     <>
//       <Header />
//       {summaryPopupIsVisible && (
//         <SummaryEditor
//           toggleResetPopup={toggleResetPopup}
//           patientInfo={patientInfo}
//           summaryID={summary.id}
//           description={summary.description}
//           sessionInfo={summary.sessionInfo}
//           setSummaryPopupIsVisible={setSummaryPopupIsVisible}
//           emotionAnalysis={summary.emotionAnalysis}
//           handleSummarySubmit={handleSummarySubmit}
//         />
//       )}

//       <div className="main-container">
//         <h1>Realtime Emotion Analysis</h1>
//         <h3>Recording Time Elapsed: {formatTime(elapsed)}</h3>
//         {!isConnected && <span>Not connected to server</span>}
//         {isConnected && timerRef.current === null && !isPaused && (
//           <Button
//             imagePath={mic}
//             text="Start Recording"
//             onClick={handleStart}
//           />
//         )}
//         {(timerRef.current !== null || isPaused) && (
//           <AudioControl
//             onPause={handlePause}
//             onResume={handleResume}
//             onEnd={handleEnd}
//             isPaused={isPaused}
//           />
//         )}
//         <h3>Dominant Emotion: {dominantEmotion}</h3>
//         <IndividualEmotionDescription emotions={emotions} />
//       </div>
//     </>
//   );
// }
import { useState, useRef, useEffect, useContext, useCallback } from "react";
import { useParams } from "react-router-dom";
import { io } from "socket.io-client";
import Header from "../components/Header";
import Button from "../components/Button";
import mic from "../assets/icons/mic-rec.png";
import AudioControl from "../components/AudioControl";
import IndividualEmotionDescription from "../components/IndividualEmotionDescription";
import { UserContext } from "../UserStore";
import SummaryEditor from "../components/SummaryEditor";

export default function Realtime() {
  const [summaryPopupIsVisible, setSummaryPopupIsVisible] = useState(false);
  const [summary, setSummary] = useState({
    id: 0,
    description: "",
    sessionInfo: null,
    emotionAnalysis: null,
  });
  const [elapsed, setElapsed] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [dominantEmotion, setDominantEmotion] = useState("Calm");
  const [currentVolume, setCurrentVolume] = useState(0);

  // Configuration for volume threshold
  const VOLUME_THRESHOLD = 0.01; // Adjust this value between 0-1 based on testing
  const SILENT_FRAMES_THRESHOLD = 5; // Number of silent frames before we consider it background noise
  const silentFramesCountRef = useRef(0);

  const { user } = useContext(UserContext);
  const timerRef = useRef(null);
  const socketRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const { email, fullname, contact } = useParams();

  const [emotions, setEmotions] = useState({
    Neutral: { val: 0, emoji: "😐" },
    Calm: { val: 0, emoji: "😌" },
    Happy: { val: 0, emoji: "😊" },
    Sad: { val: 0, emoji: "😞" },
    Angry: { val: 0, emoji: "😠" },
    Fearful: { val: 0, emoji: "😰" },
    Surprised: { val: 0, emoji: "😮" },
    Disgusted: { val: 0, emoji: "🤮" },
  });

  const patientInfo = {
    email: decodeURIComponent(email),
    fullname: decodeURIComponent(fullname),
    contact: decodeURIComponent(contact),
  };

  function toggleResetPopup() {
    setSummaryPopupIsVisible(false);
  }

  const handleSummarySubmit = (updatedSummary) => {
    setSummary((prev) => ({
      ...prev,
      description: updatedSummary,
    }));
  };

  const capitalizeFirstLetter = useCallback((string) => {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }, []);

  // Function to calculate audio volume level
  const calculateVolume = (audioData) => {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      sum += Math.abs(audioData[i]);
    }
    return sum / audioData.length;
  };

  const updateEmotionsFromResult = useCallback(
    (result) => {
      if (!result || !result.confidence_scores) return;

      // Ensure we're working with numbers, not strings
      setEmotions((prev) => ({
        Neutral: {
          ...prev.Neutral,
          val: Math.round(result.confidence_scores.neutral * 100) + "%",
        },
        Calm: {
          ...prev.Calm,
          val: Math.round(result.confidence_scores.calm * 100) + "%",
        },
        Happy: {
          ...prev.Happy,
          val: Math.round(result.confidence_scores.happy * 100) + "%",
        },
        Sad: {
          ...prev.Sad,
          val: Math.round(result.confidence_scores.sad * 100) + "%",
        },
        Angry: {
          ...prev.Angry,
          val: Math.round(result.confidence_scores.angry * 100) + "%",
        },
        Fearful: {
          ...prev.Fearful,
          val: Math.round(result.confidence_scores.fearful * 100) + "%",
        },
        Surprised: {
          ...prev.Surprised,
          val: Math.round(result.confidence_scores.surprised * 100) + "%",
        },
        Disgusted: {
          ...prev.Disgusted,
          val: Math.round(result.confidence_scores.disgust * 100) + "%",
        },
      }));

      // Log the predicted emotion for debugging
      console.log(
        `Received emotion: ${result.predicted_emotion} with scores:`,
        result.confidence_scores
      );

      setDominantEmotion(capitalizeFirstLetter(result.predicted_emotion));
    },
    [capitalizeFirstLetter]
  );

  useEffect(() => {
    const jwt = "your_jwt_token";
    socketRef.current = io("http://localhost:5000", {
      auth: { token: jwt },
    });

    socketRef.current.on("connect", () => {
      console.log("Connected");
      setIsConnected(true);
    });

    socketRef.current.on("connect_error", (error) => {
      console.log("Connection Error:", error.message);
      setIsConnected(false);
    });

    socketRef.current.on("analysis_started", (data) => {
      console.log("Session started:", data.message);
    });

    socketRef.current.on("analysis_result", (data) => {
      console.log("Analysis result:", data.result);
      updateEmotionsFromResult(data.result);
    });

    socketRef.current.on("analysis_ended", (data) => {
      console.log("Session ended:", data);
      if (data.status === "success") {
        setSummary({
          id: data.summary.save_result.analysis_id,
          description: data.summary.summary.summary_description,
          sessionInfo: data.summary.summary.session_info,
          emotionAnalysis: Object.keys(
            data.summary.summary.emotion_analysis.dominant_emotions
          )[0],
        });
        setSummaryPopupIsVisible(true);
      }
    });

    socketRef.current.on("error", (error) => {
      console.log("Error:", error.message);
    });

    socketRef.current.on("disconnect", () => {
      console.log("Disconnected");
      setIsConnected(false);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [updateEmotionsFromResult]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const secs = (seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${secs}`;
  };

  const startSession = useCallback(() => {
    const newSessionId = "session_" + Date.now();
    setSessionId(newSessionId);

    socketRef.current.emit("start_analysis", {
      session_id: newSessionId,
      patient_fullname: patientInfo.fullname,
      patient_email: patientInfo.email,
      patient_contact: patientInfo.contact,
      therapist_id: user.id,
    });

    return newSessionId;
  }, [patientInfo.fullname, patientInfo.email, patientInfo.contact, user.id]);

  const handleStart = async () => {
    try {
      const newSessionId = startSession();

      // Request audio with proper constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000, // Use 16kHz to match common ML models
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      // Use raw PCM encoding for better compatibility with ML models
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);

      source.connect(processor);
      processor.connect(audioCtx.destination);

      processor.onaudioprocess = (e) => {
        if (socketRef.current && socketRef.current.connected) {
          const inputData = e.inputBuffer.getChannelData(0);

          // Calculate volume for this frame
          const volume = calculateVolume(inputData);
          setCurrentVolume(volume);

          // Check if volume is above threshold
          if (volume > VOLUME_THRESHOLD) {
            // Reset silent frames counter if sound is detected
            silentFramesCountRef.current = 0;

            // Convert to 16-bit PCM for better compatibility
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
              pcmData[i] = inputData[i] * 32767;
            }

            // Send as base64 encoded PCM data
            const base64Data = btoa(
              String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer))
            );

            socketRef.current.emit("audio_chunk", {
              session_id: newSessionId,
              audio_data: base64Data,
              sample_rate: 16000,
              format: "pcm_16bit",
            });
            console.log("Audio sent: volume level", volume.toFixed(4));
          } else {
            // Increment silent frames counter
            silentFramesCountRef.current++;

            // Only log occasionally to avoid console spam
            if (silentFramesCountRef.current % 20 === 0) {
              console.log(
                `Audio below threshold (${volume.toFixed(
                  4
                )}), skipping. Silent for ${
                  silentFramesCountRef.current
                } frames.`
              );
            }
          }
        }
      };

      // Store the processor for cleanup
      mediaRecorderRef.current = { processor, source, audioCtx };

      timerRef.current = setInterval(() => {
        setElapsed((prevElapsed) => prevElapsed + 1);
      }, 1000);

      setIsPaused(false);
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const handlePause = () => {
    if (mediaRecorderRef.current) {
      // Disconnect the processor to pause audio processing
      mediaRecorderRef.current.source.disconnect();
    }

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      setIsPaused(true);
    }
  };

  const handleResume = () => {
    if (mediaRecorderRef.current) {
      // Reconnect the processor to resume audio processing
      mediaRecorderRef.current.source.connect(
        mediaRecorderRef.current.processor
      );
      mediaRecorderRef.current.processor.connect(
        mediaRecorderRef.current.audioCtx.destination
      );
    }

    if (timerRef.current === null && isPaused) {
      timerRef.current = setInterval(() => {
        setElapsed((prevElapsed) => prevElapsed + 1);
      }, 1000);
      setIsPaused(false);
    }
  };

  const handleEnd = () => {
    if (mediaRecorderRef.current) {
      // Clean up audio processing
      mediaRecorderRef.current.source.disconnect();
      mediaRecorderRef.current.processor.disconnect();
      mediaRecorderRef.current.audioCtx.close().catch(console.error);
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    if (socketRef.current && sessionId) {
      socketRef.current.emit("end_analysis", { session_id: sessionId });
    }

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      setElapsed(0);
      setIsPaused(false);
    }

    silentFramesCountRef.current = 0;
    setCurrentVolume(0);
    setSessionId(null);
  };

  // Helper function to render volume indicator
  const renderVolumeIndicator = () => {
    // Only show when recording
    if (!sessionId) return null;

    const volumePercentage = Math.min(currentVolume * 100, 100);
    const color = currentVolume > VOLUME_THRESHOLD ? "#4CAF50" : "#FF5722";

    return (
      <div className="volume-meter">
        <div className="volume-label">
          Volume: {(currentVolume * 100).toFixed(1)}%
          {currentVolume <= VOLUME_THRESHOLD &&
            " (Below threshold - not sending)"}
        </div>
        <div
          className="volume-bar-container"
          style={{
            width: "100%",
            backgroundColor: "#e0e0e0",
            height: "10px",
            borderRadius: "5px",
            margin: "5px 0 15px 0",
          }}
        >
          <div
            className="volume-bar"
            style={{
              width: `${volumePercentage}%`,
              backgroundColor: color,
              height: "100%",
              borderRadius: "5px",
              transition: "width 0.1s ease-in-out",
            }}
          />
        </div>
        <div
          className="threshold-marker"
          style={{ marginBottom: "10px", fontSize: "0.8rem" }}
        >
          Threshold: {(VOLUME_THRESHOLD * 100).toFixed(1)}%
        </div>
      </div>
    );
  };

  return (
    <>
      <Header />
      {summaryPopupIsVisible && (
        <SummaryEditor
          toggleResetPopup={toggleResetPopup}
          patientInfo={patientInfo}
          summaryID={summary.id}
          description={summary.description}
          sessionInfo={summary.sessionInfo}
          setSummaryPopupIsVisible={setSummaryPopupIsVisible}
          emotionAnalysis={summary.emotionAnalysis}
          handleSummarySubmit={handleSummarySubmit}
        />
      )}

      <div className="main-container">
        <h1>Realtime Emotion Analysis</h1>
        <h3>Recording Time Elapsed: {formatTime(elapsed)}</h3>
        {!isConnected && <span>Not connected to server</span>}
        {isConnected && timerRef.current === null && !isPaused && (
          <Button
            imagePath={mic}
            text="Start Recording"
            onClick={handleStart}
          />
        )}
        {(timerRef.current !== null || isPaused) && (
          <AudioControl
            onPause={handlePause}
            onResume={handleResume}
            onEnd={handleEnd}
            isPaused={isPaused}
          />
        )}

        {renderVolumeIndicator()}

        <h3>Dominant Emotion: {dominantEmotion}</h3>
        <IndividualEmotionDescription emotions={emotions} />
      </div>
    </>
  );
}
