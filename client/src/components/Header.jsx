import Logo from "./Logo";
import NavigationMenu from "./NavigationMenu";
import LoginPopup from "./LoginPopup";
import {useState} from "react";
import RegisterPopup from "./RegisterPopup";
import ProfilePopup from "./ProfilePopup";
export default function Header(){
    const [loginPopupIsVisible, setLoginPopupIsVisible] = useState(false);
    const [registerPopupIsVisible, setRegisterPopupIsVisible] = useState(false);
    const [profilePopupIsVisible, setProfilePopupIsVisible] = useState(false);

    function toggleLoginPopup() {
        setLoginPopupIsVisible(!loginPopupIsVisible);
        setRegisterPopupIsVisible(false);
    }
    function toggleRegisterPopup() {
        setRegisterPopupIsVisible(!registerPopupIsVisible);
        setLoginPopupIsVisible(false);
    }
    function toggleRestPopup() {
        setLoginPopupIsVisible(false);
        setRegisterPopupIsVisible(false);
        setProfilePopupIsVisible(false);
    }
    function toggleProfilePopup() {
        setProfilePopupIsVisible(!profilePopupIsVisible);
    }
    return(
        <header>
            <Logo />
            <NavigationMenu onGetStarted={toggleLoginPopup} onProfile={toggleProfilePopup}/>
            {loginPopupIsVisible ? <LoginPopup togglePopup={toggleRegisterPopup} toggleResetPopup={toggleRestPopup}/> : ""}
            {registerPopupIsVisible ? <RegisterPopup togglePopup={toggleLoginPopup} toggleResetPopup={toggleRestPopup}/> : ""}
            {profilePopupIsVisible ? <ProfilePopup togglePopup={toggleLoginPopup} toggleResetPopup={toggleRestPopup}/> : ""}
        </header>
    )
}