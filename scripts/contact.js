
window.onload = function() {
    emailjs.init("user_pqO3HtclZTUFDOwnE797p")

    var submitButton = document.getElementById("submit");
    document.getElementById("contact_form").addEventListener("submit", function(event) {
        event.preventDefault();
        submitButton.value = "Sending...";

        const serviceID = "default_service";
        const templateID = "contact_form";

        emailjs.sendForm(serviceID, templateID, this)
            .then(() => {
                submitButton.value = "Sent"
                alert("Email successfully sent")
                submitButton.value = "Send"
            }, (err) => {
                submitButton.value = "Send Email"
                alert(JSON.stringify(err));
                submitButton.value = "Send"
            })
    })
}