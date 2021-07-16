
// change background color
window.onload = function() {
    document.body.style.backgroundColor = "#15181b";
}


function toggleMenuBar() {
    var mobilePageContainer = document.getElementById("mobile_pages_container_1");
    if (mobilePageContainer.style.display === "block") {
        mobilePageContainer.style.display = "none";
    } else {    // if menu bar selected
        mobilePageContainer.style.display = "block";
    }
}