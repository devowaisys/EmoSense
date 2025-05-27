import Header from "../components/Header";
import PrivacyPolicyImage from "../assets/icons/privacy-policy.png"; // Importing image

export default function PrivacyPolicy() {
    return (
        <>
            <Header/>
            <div className="main-container">
                <main>
                    <div className="left-container">
                        <img src={PrivacyPolicyImage} alt="Privacy Policy" style={{width: "50%", height: '60%'}}/>
                    </div>
                    <div className="right-container">
                        <h3>
                            Privacy Policy
                        </h3>
                        <span>
                        At Emo Sense, your privacy is our top priority. We are committed to ensuring the highest level of data security and confidentiality. The only information we store includes your authentication details and session history, which are safeguarded with advanced encryption techniques. Rest assured, your data is never shared with any third parties, and we uphold strict policies to protect your personal and professional information.
                    </span>
                    </div>
                </main>
            </div>
            </>
            );
            }
