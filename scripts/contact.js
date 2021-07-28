
emailjs.init('user_pqO3HtclZTUFDOwnE797p')

window.onload = function() {
  document.getElementById('contact_form').addEventListener('submit', (event) => {
    event.preventDefault()
    // generate a five digit number for the contact_number variable
    this.contact_number.value = (Math.random() * 100000) | 0
    // these IDs from the previous steps
    emailjs.sendForm('contact_service', 'contact_form', this).then(
      function () {
        console.log('SUCCESS!')
      },
      function (error) {
        console.log('FAILED...', error)
      }
    )
  })
}
