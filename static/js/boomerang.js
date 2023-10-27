
var video = document.getElementById('eagle');
var isPlayingForward = true;
var frameTime = 1/30; // Adjust this according to your video's framerate

video.onended = function() {
    if (isPlayingForward) {
        // When the video ends, start playing backward
        isPlayingForward = false;
        video.pause();
        video.currentTime -= frameTime; // Decrement once to start the reverse playback
    }
};

video.ontimeupdate = function() {
    if (!isPlayingForward) {
        if (video.currentTime <= 0) {
            // When the backward playback finishes, start playing forward
            isPlayingForward = true;
            video.play();
        } else {
            video.currentTime -= frameTime;
        }
    }
};

// Start the video
video.play();