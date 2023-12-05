/* Fade in & scale */
$(".popup.fade-in-scale").on("click", function () {

  $.magnificPopup.open({
    items: {
      src: '#event-content',
      type: 'inline'
    },
    closeMarkup: '',
    callbacks: {
      beforeOpen: function () {
        this.st.mainClass = 'fade-in-scale';
      }
    }
  }, 0);


  /* custom close button */
  $(".closeBtn").on('click', function (e) {
    e.preventDefault();
    $(".mfp-wrap").addClass('mfp-removing');
    setTimeout(function () {
      $.magnificPopup.close();
    }, 500);
  });

});

/* Slide in (top) */
$(".popup.slide-in-top").on("click", function () {

  $.magnificPopup.open({
    items: {
      src: '#event-content',
      type: 'inline'
    },
    closeMarkup: '',
    callbacks: {
      beforeOpen: function () {
        this.st.mainClass = 'slide-in-top';
      }
    }
  }, 0);


  $(".closeBtn").on('click', function (e) {
    e.preventDefault();
    $(".mfp-wrap").addClass('mfp-removing');
    setTimeout(function () {
      $.magnificPopup.close();
    }, 500);
  });

});


/* 3D Flip (horizontal) */
$(".popup.flip-h-3d").on("click", function () {

  $.magnificPopup.open({
    items: {
      src: '#event-content',
      type: 'inline'
    },
    closeMarkup: '',
    callbacks: {
      beforeOpen: function () {
        this.st.mainClass = 'flip-h-3d';
      }
    }
  }, 0);


  $(".closeBtn").on('click', function (e) {
    e.preventDefault();
    $(".mfp-wrap").addClass('mfp-removing');
    setTimeout(function () {
      $.magnificPopup.close();
    }, 500);
  });

});

/* Rotate Carouse (left) */
$(".popup.rotate-carouse-left").on("click", function () {

  $.magnificPopup.open({
    items: {
      src: '#event-content',
      type: 'inline'
    },
    closeMarkup: '',
    callbacks: {
      beforeOpen: function () {
        this.st.mainClass = 'rotate-carouse-left';
      }
    }
  }, 0);


  $(".closeBtn").on('click', function (e) {
    e.preventDefault();
    $(".mfp-wrap").addClass('mfp-removing');
    setTimeout(function () {
      $.magnificPopup.close();
    }, 500);
  });

});