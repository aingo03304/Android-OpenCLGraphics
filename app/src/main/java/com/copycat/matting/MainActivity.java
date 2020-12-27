package com.copycat.matting;

import java.io.IOException;
import java.io.InputStream;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    int[] pixels;
    int[] trimapPixels;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        init(this.getAssets());
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        ImageView iv = findViewById(R.id.sample_image);
        try {
            Bitmap bitmapSource = loadImageFromAsset("source.png");
            pixels = new int[bitmapSource.getWidth() * bitmapSource.getHeight()];

            iv.setImageBitmap(bitmapSource);
            bitmapSource.getPixels(pixels, 0, bitmapSource.getWidth(), 0, 0, bitmapSource.getWidth(), bitmapSource.getHeight());

            Bitmap bitmapTrimap = loadImageFromAsset("trimap.png");
            iv.setImageBitmap(bitmapTrimap);
            trimapPixels = new int[bitmapTrimap.getWidth() * bitmapTrimap.getHeight()];
            bitmapTrimap.getPixels(trimapPixels, 0, bitmapTrimap.getWidth(), 0, 0, bitmapTrimap.getWidth(), bitmapTrimap.getHeight());
            drawTest(pixels, trimapPixels, bitmapSource.getHeight(), bitmapSource.getWidth());
            Bitmap alpha = Bitmap.createBitmap(trimapPixels, 0, bitmapTrimap.getWidth(), bitmapTrimap.getWidth(), bitmapTrimap.getHeight(), Bitmap.Config.ARGB_8888);
            iv.setImageBitmap(alpha);
        } catch (Exception e) {
            Log.e("MAIN_ACTIVITY", "exc");
        }
    }

    private Bitmap loadImageFromAsset(String filename) throws Exception {
        Bitmap bitmap;
        try {
            InputStream inputStream = this.getAssets().open(filename);
            bitmap = BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            Log.e("MAIN_ACTIVITY", "Failed to open the image file.");
            throw new Exception("cannot open image");
        }
        return bitmap;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native void init(AssetManager assetManager);
    public native void drawTest(int[] pixels, int[] trimapPixels, int height, int width);
}
