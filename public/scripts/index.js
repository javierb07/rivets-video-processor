let imagesPath = "images/";
let normals = [imagesPath+"1-Normal/normal1.bmp",imagesPath+"1-Normal/normal2.bmp",imagesPath+"1-Normal/normal3.bmp",imagesPath+"1-Normal/normal4.bmp",imagesPath+"1-Normal/normal5.bmp"];
let grayscales = [imagesPath+"2-Grayscale/grayscale1.bmp",imagesPath+"2-Grayscale/grayscale2.bmp",imagesPath+"2-Grayscale/grayscale3.bmp",imagesPath+"2-Grayscale/grayscale4.bmp",imagesPath+"2-Grayscale/grayscale5.bmp"];
let croppeds = [imagesPath+"3-Cropped/cropped1.bmp",imagesPath+"3-Cropped/cropped2.bmp",imagesPath+"3-Cropped/cropped3.bmp",imagesPath+"3-Cropped/cropped4.bmp",imagesPath+"3-Cropped/cropped5.bmp"];
let contrasts = [imagesPath+"4-Contrast/contrast1.bmp",imagesPath+"4-Contrast/contrast2.bmp",imagesPath+"4-Contrast/contrast3.bmp",imagesPath+"4-Contrast/contrast4.bmp",imagesPath+"4-Contrast/contrast5.bmp"];
let blackwhites = [imagesPath+"5-BlackWhite/blackwhite1.bmp",imagesPath+"5-BlackWhite/blackwhite2.bmp",imagesPath+"5-BlackWhite/blackwhite3.bmp",imagesPath+"5-BlackWhite/blackwhite4.bmp",imagesPath+"5-BlackWhite/blackwhite5.bmp"];
let resizeds = [imagesPath+"6-Resized/resized1.bmp",imagesPath+"6-Resized/resized2.bmp",imagesPath+"6-Resized/resized3.bmp",imagesPath+"6-Resized/resized4.bmp",imagesPath+"6-Resized/resized5.bmp"];
let outputs = [imagesPath+"7-Output/output1.bmp",imagesPath+"7-Output/output2.bmp",imagesPath+"7-Output/output3.bmp",imagesPath+"7-Output/output4.bmp",imagesPath+"7-Output/output5.bmp"];
let targets = [imagesPath+"8-Target/target1.bmp",imagesPath+"8-Target/target2.bmp",imagesPath+"8-Target/target3.bmp",imagesPath+"8-Target/target4.bmp",imagesPath+"8-Target/target5.bmp"];

let x = 0;

function displayNextImage(images, stage) {
    $(stage).attr("src", images[x]);
}
function incrementCounter(){
    if(x >= normals.length-1){
        x=0;
    } else {
        x++;
    }
}

$( document ).ready(function() {
    let height = $("#normals").height();
    $("img").attr('height',height);
    let timer = 500;
    setInterval(() => incrementCounter(), timer);
    setInterval(() => displayNextImage(normals,"#normals"), timer);
    setInterval(() => displayNextImage(grayscales,"#grayscales"), timer);
    setInterval(() => displayNextImage(croppeds,"#croppeds"), timer);
    setInterval(() => displayNextImage(contrasts,"#contrasts"), timer);
    setInterval(() => displayNextImage(blackwhites,"#blackwhites"), timer);
    setInterval(() => displayNextImage(resizeds,"#resizeds"), timer);
    setInterval(() => displayNextImage(outputs,"#outputs"), timer);
    setInterval(() => displayNextImage(targets,"#targets"), timer);
});