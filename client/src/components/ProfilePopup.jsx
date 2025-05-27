import Button from "./Button";
import Icon from "./Icon";
import profileUserIcon from "../assets/icons/cancel.png";
import { useContext, useState } from "react";
import { UserContext } from "../UserStore";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function ProfilePopup({ toggleResetPopup }) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");
  const { removeUser } = useContext(UserContext);
  const navigate = useNavigate();
  const { user, accessToken } = useContext(UserContext);

  async function handleLogout() {
    // TODO Implement a popup for session timeout
    const token = localStorage.getItem("accessToken");
    console.log("Token:", token);

    if (!token) {
      removeUser();
      navigate("/");
      return;
    }

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/logout_therapist",
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          timeout: 10000,
        }
      );

      if (response.data.success === true) {
        localStorage.removeItem("accessToken"); // Clear the token
        removeUser();
        setError("");
        navigate("/");
      } else {
        setError(response.data.message || "An error occurred during logout.");
      }
    } catch (err) {
      if (err.response?.status === 401) {
        let countdown = 3; // Starting countdown value
        setError(
          `Your session has expired. Redirecting in ${countdown} seconds...`
        );

        const interval = setInterval(() => {
          countdown -= 1;
          if (countdown > 0) {
            setError(
              `Your session has expired. Redirecting in ${countdown} seconds...`
            );
          }
        }, 1000); // Update message every second

        setTimeout(() => {
          clearInterval(interval); // Stop the countdown updates
          localStorage.removeItem("accessToken"); // Clear the token
          removeUser();
          navigate("/");
        }, 3000); // Redirect after 3 seconds

        return;
      }

      // Handle other errors
      if (err.response) {
        setError(
          err.response.data?.message || "An error occurred on the server."
        );
      } else if (err.request) {
        setError("Server did not respond. Please try again.");
      } else if (err.code === "ECONNABORTED") {
        setError("Request timed out. Please try again.");
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    }
  }

  async function handleSubmit(evt) {
    evt.preventDefault();
    setError("");
    setMsg("");

    if (!email.includes("@") || !email.endsWith(".com")) {
      setError("Invalid email");
      return;
    }

    try {
      const response = await axios.put(
        `http://127.0.0.1:5000/api/update_therapist/${user.id}`,
        {
          full_name: fullName,
          email: email,
          curr_password: currentPassword,
          new_password: newPassword,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          timeout: 10000,
        }
      );

      if (response.data.success === true) {
        setError("");
        setMsg(response.data.message);
      } else {
        setMsg("");
        setError(response.data.message || "Update failed.");
      }
    } catch (err) {
      if (err.response) {
        setError(
          err.response.data?.message || "An error occurred on the server."
        );
      } else if (err.request) {
        setError("Server did not respond. Please try again.");
      } else if (err.code === "ECONNABORTED") {
        setError("Request timed out. Please try again.");
      } else {
        setError(err.message);
      }
    }
  }

  return (
    <div className={"popup-outer-container"}>
      <Icon
        path={profileUserIcon}
        customStyle={{ position: "fixed", top: 30 }}
        onClick={toggleResetPopup}
      />
      <h1>Update Profile</h1>
      {msg ? (
        <span className={"success-message"}>{msg}</span>
      ) : error ? (
        <span className={"err-msg"}>{error}</span>
      ) : null}
      <form className={"popup-inner-form"}>
        <h3>Full Name</h3>
        <input
          type={"text"}
          id={"name"}
          placeholder={user.full_name || ""}
          onChange={(evt) => setFullName(evt.target.value)}
        />
        <h3>Email</h3>
        <input
          type={"email"}
          id={"email"}
          placeholder={user.email || ""}
          onChange={(evt) => setEmail(evt.target.value)}
        />
        <h3>Current Password</h3>
        <input
          type={"password"}
          id={"curr-password"}
          onChange={(evt) => setCurrentPassword(evt.target.value)}
        />
        <h3>New Password</h3>
        <input
          type={"password"}
          id={"new-password"}
          onChange={(evt) => setNewPassword(evt.target.value)}
        />
        <div className={"button-container"}>
          <Button width={"13rem"} text={"Update"} onClick={handleSubmit} />
          <Button width={"13rem"} text={"Logout"} onClick={handleLogout} />
        </div>
      </form>
    </div>
  );
}
