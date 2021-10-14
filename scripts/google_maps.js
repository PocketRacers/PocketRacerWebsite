
function initMap() {
    // coord locations of Engineering IV
    const labLocation = { lat: 34.068923803985506, lng: -118.44420989246592 };    
    const map = new google.maps.Map(document.getElementById("map"), {
          zoom: 13.5,
          center: labLocation,
          mapId: "72bd48810ad41574"
    });

    // The marker, positioned at Engineering IV 
    const marker = new google.maps.Marker({
        position: labLocation,
        map: map,
    });
}
