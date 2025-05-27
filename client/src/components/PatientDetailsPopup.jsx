import Button from "./Button";
import Icon from "./Icon";
import cancel from "../assets/icons/cancel.png";
import { useState } from "react";

export default function PatientDetailsPopup({
  fullname,
  email,
  contact,
  setFullName,
  setEmail,
  setContact,
  toggleResetPopup,
  mode,
  onSaveSuccess,
}) {
  const [error, setError] = useState("");
  const [contactError, setContactError] = useState("");

  // Phone number validation function
  const validatePhoneNumber = (phone) => {
    // Basic international phone number regex
    const phoneRegex = /^[+]?[(]?[0-9]{1,4}[)]?[-\s./0-9]*$/;

    // Remove all non-digit characters for length check
    const digitsOnly = phone.replace(/\D/g, "");

    // Check if empty
    if (!phone) return "Phone number is required";

    // Check format validity
    if (!phoneRegex.test(phone)) return "Invalid phone number format";

    // Check minimum length (7 is the shortest valid phone number - e.g. in some countries)
    if (digitsOnly.length < 7) return "Phone number too short";

    // Check maximum length (15 digits is E.164 standard max)
    if (digitsOnly.length > 15) return "Phone number too long";

    return "";
  };

  function handleSubmit(evt) {
    evt.preventDefault();
    setError("");
    setContactError("");

    // Validate all fields
    if (fullname === "" || email === "" || contact === "") {
      setError("Fields cannot be empty.");
      return;
    }

    if (!email.includes("@") || !email.includes(".")) {
      setError("Invalid email");
      return;
    }

    // Validate phone number
    const phoneValidationError = validatePhoneNumber(contact);
    if (phoneValidationError) {
      setContactError(phoneValidationError);
      return;
    }

    // Close popup and trigger success callback
    toggleResetPopup();
    if (onSaveSuccess) {
      onSaveSuccess(mode);
    }
  }

  // Format phone number as user types
  const handlePhoneChange = (evt) => {
    let value = evt.target.value;

    // Allow only numbers, +, -, (, ), and spaces
    value = value.replace(/[^0-9+() -]/g, "");

    // Limit length to 20 characters (including formatting)
    if (value.length > 20) {
      return;
    }

    setContact(value);
    // Clear error when user starts typing
    if (contactError && value) {
      setContactError("");
    }
  };

  return (
    <div className="popup-outer-container">
      <Icon
        path={cancel}
        customStyle={{ position: "fixed", top: 30 }}
        onClick={toggleResetPopup}
      />
      <h1>Patient Details</h1>
      {error && <span className="err-msg">{error}</span>}
      <form className="popup-inner-form">
        <h3>Full Name</h3>
        <input
          type="text"
          id="fullname"
          value={fullname}
          onChange={(evt) => setFullName(evt.target.value)}
          required
        />

        <h3>Email</h3>
        <input
          type="email"
          id="email"
          value={email}
          onChange={(evt) => setEmail(evt.target.value)}
          required
        />

        <h3>Contact</h3>
        <input
          type="tel"
          id="contact"
          value={contact}
          onChange={handlePhoneChange}
          placeholder="+1 (123) 456-7890"
          pattern="[0-9+() -]*"
          maxLength="20"
          required
        />
        {contactError && <span className="err-msg">{contactError}</span>}

        <div className="button-container">
          <Button width="13rem" text="Save" onClick={handleSubmit} />
        </div>
      </form>
    </div>
  );
}
