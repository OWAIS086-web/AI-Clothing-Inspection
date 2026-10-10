$(document).ready(function() {
    const socket = io();

    socket.on('finalresults', function(data) {
        $(".loaderDiv").addClass("d-none");
        const decodedText = atob(data.logs);
        $("#time").text(data.time_taken);

        // Clear the old content
        $("#pdf_display").html('');
        $("#listimg").html('');
        $("#listimg2").html('');
        $("#logs").html(decodedText.replace(/\n/g, '<br>'));

        // Display PDF images in Owl Carousel
        displayImages('listimg', 'deflist', data.pdf_image);

        // Display Defect List images in Owl Carousel
        displayImages('listimg2', 'deflist2', data.defect_list);

        $(".pages").removeClass('d-none');
        $(".pages2").removeClass('d-none');
    });

    function displayImages(listId, carouselId, imageData) {
        let itm = '';
        imageData.forEach((imgData, index) => {
            if (imgData) {
                itm += `
                    <div class="owl-item">
                        <img class="img-fluid" src="data:image/png;base64,${imgData}" alt="Image ${index + 1}">
                    </div>`;
            } else {
                console.error('Invalid image data at index ' + index);
            }
        });

        $(`#${listId}`).html(itm);

        $(`#${carouselId}`).owlCarousel({
            loop: true,
            margin: 10,
            dots: false,
            responsiveClass: true,
            onInitialized: carouselCounter,
            onChanged: carouselCounter,
            responsive: {
                0: {
                    items: 1,
                    nav: true
                },
                600: {
                    items: 1,
                    nav: false
                },
                1000: {
                    items: 1,
                    nav: true,
                    loop: false
                }
            }
        });

        $(`#${carouselId}`).trigger('refresh.owl.carousel');
    }

    function carouselCounter(event) {
        var items = event.item.count;
        var item = event.item.index + 1;
        if (item > items) {
            item = item - items;
        }
        $('#counter').html(item + " of " + items);
    }
    
    function base64ToText(base64) {
        const binaryString = atob(base64);
        const binaryLen = binaryString.length;
        const bytes = new Uint8Array(binaryLen);
        for (let i = 0; i < binaryLen; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const decoder = new TextDecoder();
        return decoder.decode(bytes);
    }
});
