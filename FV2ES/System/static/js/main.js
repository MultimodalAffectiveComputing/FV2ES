$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    document.getElementByID
    // upload video
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                //change here
                var objUrl = reader.readAsDataURL(input.files[0]);
                console.log("objUrl = "+objUrl);
                console.log('line 15');
                $("#videoPreview").attr("src", objUrl);
            }
        }
    }
    function getObjectURL(file) {
         var url = null ;
         if (window.createObjectURL!=undefined) { // basic
             url = window.createObjectURL(file) ;
             console.log('1')}
//       else if(window.navigator && window.navigator.msSaveOrOpenBlob){
//            console.log(file.name)
//            url = '"' + 'http://127.0.0.1/5000/upload/' + file.name + '"'
////            url = "'" + url + "'"
////            url = '"Ses01F_impro01.avi"'
//            console.log('4')
//         }
         else if (window.URL!=undefined) { // mozilla(firefox)
             url = window.URL.createObjectURL(file)
             console.log('2')
         } else if (window.webkitURL!=undefined) { // webkit or chrome
             url = window.webkitURL.createObjectURL(file) ;
             console.log('3')
         }
         return url ;
    }
    $("#imageUpload").change(function () {
        // show UI
        $('.image-section').show();
        $('#btn-predict').show();
        // objurl to get and add video source
        console.log(this.files[0])
        var objUrl = getObjectURL(this.files[0]) ;
             console.log("objUrl = "+objUrl) ;
             if (objUrl) {
                 $("#videoPreview").attr("src", objUrl) ; // add video source
                 console.log($("#videoPreview").attr("src"))
//                 $(' <embed id="avi_preview" width="800" height="600" border="0" showdisplay="0" showcontrols="1" autostart="0" autorewind="0" playcount="0" moviewindowheight="240" moviewindowwidth="320" src= ' + objUrl + '>').appendTo('#player')

                 console.log('Finished adding video source')
                 }

    });

    // Forecast section
    $('#btn-predict').click(function (){
        // upload data
//        var form = document.getElementById('upload-file')
//        var form_data = new FormData(form);
//        console.log(form_data)
        // Show loading animation
        $('.loader').show();
        $.ajax({
            type: 'POST',
            url: '/predict',
            // data ，data sent to the server
            data: 'test',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                console.log('Success!');
                console.log(data)
                // data = JSON.parse(data);
                if(data) {
                    run(data)
                    $('.emotionbar').show()
                    }
            },

        });

    });
    // Responsible for displaying the progress bar and prompting the final emotional judgment result
    // Input is string type
    function run(data){
        // get the element object
        var angrybar = document.getElementById("angrybar");
        var excitedbar = document.getElementById("excitedbar");
        var frustratedbar = document.getElementById("frustratedbar");
        var happybar = document.getElementById("happybar");
        var neuralbar = document.getElementById("neuralbar");
        var sadbar = document.getElementById("sadbar");

        // Read the result
        data['angry'] = Number(data['angry']) * 100
        data['excited'] = Number(data['excited']) * 100
        data['frustrated'] = Number(data['frustrated']) * 100
        data['happy'] = Number(data['happy']) * 100
        data['neural'] = Number(data['neural']) * 100
        data['sad'] = Number(data['sad']) * 100
        // Judging the output
        console.log(data)
        console.log(Object.keys(data))
        var sorted_keys_array = Object.keys(data).sort((a,b)=>{
       return data[b]-data[a];
});
        result = sorted_keys_array[0] //

         $("<span style=\"font-family:'Wingdings 2';color:#FFFFF0\">R</span>").appendTo('#' + result)

        angrybar.style.width= data['angry'] + "%";
        angrybar.innerHTML = angrybar.style.width;
        excitedbar.style.width= data['excited'] + "%";
        excitedbar.innerHTML = excitedbar.style.width;
        frustratedbar.style.width= data['frustrated'] + "%";
        frustratedbar.innerHTML = frustratedbar.style.width;
        happybar.style.width= data['happy'] + "%";
        happybar.innerHTML = happybar.style.width;
        neuralbar.style.width= data['neural'] + "%";
        neuralbar.innerHTML = neuralbar.style.width;
        sadbar.style.width= data['sad'] + "%";
        sadbar.innerHTML = sadbar.style.width;

    }
});