import Header from "../components/Header";
import AboutUsImage from "../assets/icons/about-us.png"; // Importing image

export default function AboutUs() {
    return (
        <>
            <Header/>
            <div className="main-container">
                <main>
                    <div className="left-container">
                        <img src={AboutUsImage} alt="About Us"/>
                    </div>
                    <div className="right-container">
                        <h3>
                            About Us
                        </h3>
                        <span>
                        Emo Sense is dedicated to empowering therapists with real-time emotion analysis tools for more effective and insightful therapy sessions. Our platform combines innovation and simplicity to enhance the therapeutic process while maintaining a strong commitment to privacy and security.
                    </span>
                    </div>
                </main>
            </div>
            </>
            );
            }
