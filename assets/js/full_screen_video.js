document.addEventListener('DOMContentLoaded', function () {
    const videoElement = document.getElementById('bg-video');

    // When the video metadata is loaded
    videoElement.addEventListener('loadedmetadata', function () {
        // Start playing the video
        // videoElement.play();
        // Gradually fade in the video after it finishes loading
        videoElement.style.opacity = 1.0;
    });
});

// window.addEventListener("scroll", function () {
//     let imageDiv = document.querySelector('.full-page-image');
//     const videoElement = document.getElementById('bg-video');
//
//     if (window.scrollY > 75) {
//         videoElement.style.opacity = 0;
//     }
//     // Check if page is scrolled
//     if (window.scrollY > 200) { // You can adjust this value based on your requirement
//         imageDiv.classList.add('banner-state');
//         imageDiv.classList.add('no-content');
//         // imageDiv.style.backgroundImage = `url(assets/img/full_screen_2.png)`;
//     }
//     if (window.scrollY > 350) {
//         imageDiv.remove()
//     }
//     ;
// });
