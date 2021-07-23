
function initMap() {
    // coord locations of Engineering IV
    const labLocation = { lat: 34.069632890358406, lng: -118.4445256020825 };    
    const map = new google.maps.Map(document.getElementById("map"), {
          zoom: 4,
          center: labLocation,
    });

    // The marker, positioned at Engineering IV 
    const marker = new google.maps.Marker({
        position: labLocation,
        map: map,
    });
}
