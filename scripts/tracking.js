/*
    if opt in, create local stroage cookie item
    check if that item exists
    if it does, enable tracking
    otherwise, do nothing
    call this function in all webpages
*/

var disableStr = "ga-disable-UA-199894373-1" 

function checkAndTrack() {
    if (localStorage.getItem("start_tracking") === "true") {
        window[disableStr] = false
        window.dataLayer = window.dataLayer || []
        function gtag(){dataLayer.push(arguments)}
        gtag("js", new Date())
        gtag("config", "UA-199894373-1")
        alert("called")
    }
}

window.onload = function() {
    checkAndTrack()
}