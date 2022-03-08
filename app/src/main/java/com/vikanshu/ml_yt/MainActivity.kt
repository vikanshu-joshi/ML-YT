package com.vikanshu.ml_yt

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import com.vikanshu.ml_yt.databinding.ActivityMainBinding
import com.vikanshu.ml_yt.ml.Mobilenet
import com.vikanshu.ml_yt.ml.ObjectDetection
import com.vikanshu.ml_yt.ml.Segmenter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import timber.log.Timber
import java.nio.ByteBuffer
import java.util.*


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main)

        binding.button.setOnClickListener {
            startActivityForResult(Intent().apply {
                action = Intent.ACTION_GET_CONTENT
                type = "image/*"
            }, 1234)
        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val bm = MediaStore.Images.Media.getBitmap(contentResolver, data?.data)
        bitmap = Bitmap.createScaledBitmap(
            bm,
            224,
            224,
            true
        )
        binding.imageView.setImageBitmap(bitmap)
        objectDetection()

//        segmentImage()
    }
    // object detection model
    private fun objectDetection() {
        val model = ObjectDetection.newInstance(this)
        val image = TensorImage.fromBitmap(bitmap)
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray
        val classes_list =
            application.assets.open("label_map.txt").bufferedReader().use { it.readText() }
                .split("\n")
        val canvas = Canvas(bitmap)
        val p = Paint()
        p.style = Paint.Style.STROKE
        p.strokeWidth = 2f
        p.isAntiAlias = true
        p.isFilterBitmap = true
        p.isDither = true
        p.color = Color.RED
        for (i in 0 until numberOfDetections[0].toInt()) {
            if (scores[i] > 0.5) {
                val t = locations[0 + i * 4] * 100
                val d = locations[2 + i * 4] * 100
                val l = locations[1 + i * 4] * 100
                val r = locations[3 + i * 4] * 100
                val h = bitmap.height
                val w = bitmap.width
                binding.textView.text = binding.textView.text.toString() + "\nPrediction: ${classes_list[classes[i].toInt()]} \n Probablity: ${scores[i] * 100}"
                canvas.drawRect(l, t, w - r, h - d, p)
            }
        }
        binding.imageView.setImageBitmap(bitmap)
        model.close()
    }
    // image segmentation model
    private fun segmentImage() {
        val model = Segmenter.newInstance(this)
        val image = TensorImage.fromBitmap(bitmap)
        val outputs = model.process(image)
        val segmentationMasks = outputs.segmentationMasksAsCategoryList

        Timber.e("segmentationMasks size: ${segmentationMasks.size}")

        for (i in segmentationMasks) {
            Timber.e("class: ${i.displayName}")
            Timber.e("label: ${i.label}")
            Timber.e("score: ${i.score}")
        }

        model.close()
    }
}