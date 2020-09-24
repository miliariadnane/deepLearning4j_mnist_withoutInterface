import java.util.Random;

public class Test {
    public static void main(String[] args) {
        Random random = new Random(1234);
        System.out.println(random.nextInt(20));
        System.out.println(random.nextInt(20));
        System.out.println(random.nextInt(20));

    }
}
