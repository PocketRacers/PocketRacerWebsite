
window.onload = function() {
    var submitButton = document.getElementById("submit");

    alert("called 1")

    document.getElementById("contact_form").addEventListener("submit", function(event) {
        alert("called 2")

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