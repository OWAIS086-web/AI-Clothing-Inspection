$(document).ready(function() {
    $.ajax({
        url: '/api/inspections', // Endpoint to fetch inspection data
        method: 'GET',
        success: function(data) {
            const inspectionList = $('#inspection-list');
            inspectionList.empty(); // Clear existing data

            data.inspections.forEach(item => {
                // Ensure item.articleNumber is correctly handled
                let articleNumber = item.articleNumber || 'No filename available';

                // Create defect images HTML for Owl Carousel
                let defectImagesHtml = item.defect_images.map((image, index) => `
                    <div class="item">
                        <img src="data:image/png;base64,${image}" class="style-113" alt="Defect Image ${index + 1}" />
                    </div>
                `).join('');

                let itemHtml = `
                    <div class="style-86">
                        <div class="style-87">
                            <div class="style-88">
                                <div class="style-89">
                                    <div class="style-90">
                                        <!-- Replace the existing img with defect images carousel -->
                                        <div class="owl-carousel defect-carousel">
                                            ${defectImagesHtml}
                                        </div>
                                    </div>
                                    <div class="style-92">
                                        <a href="/manager/inspectionview/${item.id}" class="style-93">
                                            <span class="style-94">Article Number</span><span class="style-95">${articleNumber}</span>
                                        </a>
                                        <div class="style-96">
                                            <p class="style-97">Article No:</p>
                                            <p class="style-98">${articleNumber}</p>
                                        </div>
                                        <div class="style-99">
                                            <p class="style-100">Inspection Date:</p>
                                            <p class="style-101">${new Date(item.created_at).toLocaleDateString()}</p>
                                        </div>
                                        <div class="style-102">
                                            <p class="style-103">Inspection Type:</p>
                                            <p class="style-104">${item.inspection_type}</p>
                                        </div>
                                        <div class="style-103">
                                            <p class="style-104">Time Taken:</p>
                                            <p class="style-105">${item.time_taken}</p>
                                        </div>
                                    </div>
                                    <div class="style-105">
                                        <h3 class="style-106">${item.defect_count}</h3>
                                        <p class="style-107">${item.defect_status}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                inspectionList.append(itemHtml);
            });

            // Initialize Owl Carousel for defect images after adding content to DOM
            $('.owl-carousel').owlCarousel({
                items: 1,
                loop: true,
                margin: 10,
                nav: true
            });
        },
        error: function(error) {
            console.error('Error fetching inspection data:', error);
        }
    });
});
