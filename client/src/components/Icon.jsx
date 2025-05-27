export default function Icon({ path, customStyle, onClick }) {
    const style = { background: `url(${path}) no-repeat center / contain` }
    return (
        <div
            className="icon"
            style={{...style, ...customStyle}}
            onClick={onClick}
        ></div>
    );
}
